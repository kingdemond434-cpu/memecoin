//! Yellowstone received, decoded and filtered without touching Python.
//!
//! The desk's earliest information used to enter through `grpc.aio`: a Python
//! socket, a Python protobuf object per update, a `WhichOneof`, a Python
//! dispatch, and a dict built for every transaction on the chain -- thousands
//! a second, almost all of which the desk immediately discards. The work of
//! discarding them was itself being done in the interpreter that has to
//! decide.
//!
//! Here the whole receive path is Rust:
//!
//!     socket -> HTTP/2 -> prost decode -> program filter -> discriminator
//!            -> HotEvent -> bounded queue
//!
//! and Python sees only what survived, in batches, as compact tuples. On a
//! stream where one transaction in several hundred is interesting, that is
//! the difference between building three thousand Python objects a second and
//! building ten.
//!
//! Three decisions worth stating, because each could reasonably have gone the
//! other way:
//!
//! **Pubkeys stay binary.** A base58 encode is ~40 bytes of allocation and a
//! division loop per key, and a transaction carries dozens. They cross as
//! `bytes` and are encoded only for the handful the desk actually acts on.
//!
//! **Dedupe happens here.** The same transaction arrives from every feed that
//! is racing. Rejecting a duplicate before it becomes a Python object is
//! the entire value of rejecting it early.
//!
//! **This is not authoritative.** It runs beside the Python client and has to
//! agree with it before anything depends on it, exactly as the Rust
//! transaction builder had to reach byte parity before it was trusted. A
//! faster path that silently drops one launch in a thousand is worse than the
//! slower path that drops none.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;

pub mod pb {
    #![allow(clippy::all, unused_qualifications)]
    // The generated geyser code refers to its dependency as
    // `super::solana::storage::confirmed_block`, so the module tree has to
    // match the proto package names exactly rather than be arranged for
    // convenience.
    pub mod solana {
        pub mod storage {
            pub mod confirmed_block {
                #![allow(clippy::all, unused_qualifications)]
                include!(concat!(
                    env!("OUT_DIR"),
                    "/solana.storage.confirmed_block.rs"
                ));
            }
        }
    }
    pub mod geyser {
        #![allow(clippy::all, unused_qualifications)]
        include!(concat!(env!("OUT_DIR"), "/geyser.rs"));
    }
    pub use geyser::*;
}

/// What survives the filter. Fixed shape, no allocation beyond the keys.
#[derive(Clone, Debug)]
pub struct HotEvent {
    /// Wall-clock nanoseconds when the update left the network stack. Taken
    /// as early as possible so the number measures the wire and not our
    /// scheduling.
    pub received_ns: u128,
    pub slot: u64,
    /// Transaction signature, 64 bytes, binary.
    pub signature: Vec<u8>,
    /// The program whose instruction matched.
    pub program: [u8; 32],
    /// First eight bytes of that instruction's data.
    pub discriminator: [u8; 8],
    /// Account keys of the matched instruction, in order, binary.
    pub accounts: Vec<[u8; 32]>,
    /// The transaction's fee payer.
    pub fee_payer: [u8; 32],
    /// The matched instruction's full data.
    pub data: Vec<u8>,
    pub is_vote: bool,
}

#[derive(Default)]
pub struct Stats {
    pub updates: AtomicU64,
    pub transactions: AtomicU64,
    pub matched: AtomicU64,
    pub duplicates: AtomicU64,
    pub dropped: AtomicU64,
    pub delivered: AtomicU64,
    pub reconnects: AtomicU64,
}

/// A bounded queue plus the recent-signature set that makes racing safe.
pub struct Sink {
    queue: Mutex<VecDeque<HotEvent>>,
    seen: Mutex<(VecDeque<[u8; 8]>, std::collections::HashSet<[u8; 8]>)>,
    capacity: usize,
    seen_capacity: usize,
    pub stats: Stats,
    pub running: AtomicBool,
    pub last_error: Mutex<String>,
}

impl Sink {
    pub fn new(capacity: usize, seen_capacity: usize) -> Self {
        Self {
            queue: Mutex::new(VecDeque::with_capacity(capacity)),
            seen: Mutex::new((
                VecDeque::with_capacity(seen_capacity),
                std::collections::HashSet::with_capacity(seen_capacity),
            )),
            capacity,
            seen_capacity,
            stats: Stats::default(),
            running: AtomicBool::new(false),
            last_error: Mutex::new(String::new()),
        }
    }

    /// True when this signature has not been seen. Keyed on the first eight
    /// bytes: a Solana signature is 64 bytes of Ed25519 output, so eight of
    /// them is 2^64 of key space against a window of a few thousand -- the
    /// collision probability over any realistic window is far below the rate
    /// at which the stream itself drops updates, and storing the full 64
    /// would triple the memory of the hot set for nothing.
    fn first_sight(&self, signature: &[u8]) -> bool {
        if signature.len() < 8 {
            return true;
        }
        let mut key = [0u8; 8];
        key.copy_from_slice(&signature[..8]);
        let mut guard = self.seen.lock().unwrap();
        if guard.1.contains(&key) {
            return false;
        }
        guard.0.push_back(key);
        guard.1.insert(key);
        while guard.0.len() > self.seen_capacity {
            if let Some(old) = guard.0.pop_front() {
                guard.1.remove(&old);
            }
        }
        true
    }

    fn push(&self, event: HotEvent) {
        let mut queue = self.queue.lock().unwrap();
        if queue.len() >= self.capacity {
            // Oldest out. A full queue means Python is behind, and in that
            // state the freshest launch is the one worth keeping.
            queue.pop_front();
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
        }
        queue.push_back(event);
    }

    /// Take up to `max` events. Bounded so one drain cannot hold the caller.
    pub fn drain(&self, max: usize) -> Vec<HotEvent> {
        let mut queue = self.queue.lock().unwrap();
        let take = max.min(queue.len());
        let out: Vec<HotEvent> = queue.drain(..take).collect();
        self.stats
            .delivered
            .fetch_add(out.len() as u64, Ordering::Relaxed);
        out
    }

    pub fn depth(&self) -> usize {
        self.queue.lock().unwrap().len()
    }
}

fn now_ns() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

fn to_key(raw: &[u8]) -> Option<[u8; 32]> {
    if raw.len() != 32 {
        return None;
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(raw);
    Some(key)
}

/// Pull every matching instruction out of one transaction update.
///
/// Both outer instructions and inner (CPI) instructions, because Pump's own
/// trade event is emitted through a CPI and a decoder that only walks the
/// outer list sees a create and never a trade.
pub fn extract(
    update: &pb::SubscribeUpdateTransaction,
    programs: &[[u8; 32]],
    received_ns: u128,
) -> Vec<HotEvent> {
    let mut out = Vec::new();
    let info = match &update.transaction {
        Some(info) => info,
        None => return out,
    };
    let transaction = match &info.transaction {
        Some(transaction) => transaction,
        None => return out,
    };
    let message = match &transaction.message {
        Some(message) => message,
        None => return out,
    };

    // Account keys, plus anything the loaded address tables contributed --
    // in the order the runtime resolves them, or an index past the static
    // keys resolves to the wrong account.
    let mut keys: Vec<[u8; 32]> = Vec::with_capacity(message.account_keys.len() + 8);
    for raw in &message.account_keys {
        if let Some(key) = to_key(raw) {
            keys.push(key);
        }
    }
    if let Some(meta) = &info.meta {
        for raw in &meta.loaded_writable_addresses {
            if let Some(key) = to_key(raw) {
                keys.push(key);
            }
        }
        for raw in &meta.loaded_readonly_addresses {
            if let Some(key) = to_key(raw) {
                keys.push(key);
            }
        }
    }
    if keys.is_empty() {
        return out;
    }
    let fee_payer = keys[0];

    let mut consider = |program_index: u32, accounts: &[u8], data: &[u8]| {
        let program = match keys.get(program_index as usize) {
            Some(key) => *key,
            None => return,
        };
        if !programs.iter().any(|candidate| *candidate == program) {
            return;
        }
        if data.len() < 8 {
            return;
        }
        let mut discriminator = [0u8; 8];
        discriminator.copy_from_slice(&data[..8]);
        let resolved: Vec<[u8; 32]> = accounts
            .iter()
            .filter_map(|index| keys.get(*index as usize).copied())
            .collect();
        out.push(HotEvent {
            received_ns,
            slot: update.slot,
            signature: info.signature.clone(),
            program,
            discriminator,
            accounts: resolved,
            fee_payer,
            data: data.to_vec(),
            is_vote: info.is_vote,
        });
    };

    for instruction in &message.instructions {
        consider(
            instruction.program_id_index,
            &instruction.accounts,
            &instruction.data,
        );
    }
    if let Some(meta) = &info.meta {
        for inner in &meta.inner_instructions {
            for instruction in &inner.instructions {
                consider(
                    instruction.program_id_index,
                    &instruction.accounts,
                    &instruction.data,
                );
            }
        }
    }
    out
}

/// Connect, subscribe and pump matching events into the sink until stopped.
pub async fn run(
    endpoint: String,
    token: Option<String>,
    programs: Vec<[u8; 32]>,
    sink: Arc<Sink>,
) {
    use pb::geyser_client::GeyserClient;

    while sink.running.load(Ordering::Relaxed) {
        match connect_and_stream(&endpoint, token.as_deref(), &programs, &sink).await {
            Ok(()) => {}
            Err(error) => {
                *sink.last_error.lock().unwrap() = error;
                sink.stats.reconnects.fetch_add(1, Ordering::Relaxed);
            }
        }
        if !sink.running.load(Ordering::Relaxed) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
    let _ = std::mem::size_of::<GeyserClient<tonic::transport::Channel>>();
}

async fn connect_and_stream(
    endpoint: &str,
    token: Option<&str>,
    programs: &[[u8; 32]],
    sink: &Arc<Sink>,
) -> Result<(), String> {
    use pb::geyser_client::GeyserClient;

    // A scheme is required. `host:port` alone parses as a URI with the host
    // in the PATH and no authority, and tonic then fails to connect to
    // something that looks superficially fine in the error message.
    let uri = if endpoint.contains("://") {
        endpoint.to_string()
    } else {
        format!("https://{endpoint}")
    };
    let secure = uri.starts_with("https://");
    let mut builder = tonic::transport::Endpoint::from_shared(uri.clone())
        .map_err(|e| format!("bad endpoint {uri}: {e}"))?
        .tcp_nodelay(true)
        .http2_adaptive_window(true)
        .connect_timeout(std::time::Duration::from_secs(10));
    if secure {
        // Explicit, because tonic does NOT infer TLS from the scheme. An
        // https:// endpoint without this fails with a bare "transport
        // error" that says nothing about the cause -- which is exactly how
        // this shipped reconnecting six times against a healthy public
        // endpoint the Python client was streaming from.
        let tls = tonic::transport::ClientTlsConfig::new().with_native_roots();
        builder = builder
            .tls_config(tls)
            .map_err(|e| format!("tls: {e}"))?;
    }
    let channel = builder
        .connect()
        .await
        .map_err(|e| format!("connect to {uri}: {e}"))?;

    let owned_token = token.map(|value| value.to_string());
    let mut client = GeyserClient::with_interceptor(
        channel,
        move |mut request: tonic::Request<()>| {
            if let Some(value) = &owned_token {
                if let Ok(parsed) = value.parse() {
                    request.metadata_mut().insert("x-token", parsed);
                }
            }
            Ok(request)
        },
    );

    let mut filters = std::collections::HashMap::new();
    filters.insert(
        "desk".to_string(),
        pb::SubscribeRequestFilterTransactions {
            vote: Some(false),
            failed: Some(false),
            account_include: programs
                .iter()
                .map(|key| bs58::encode(key).into_string())
                .collect(),
            // Everything else default: the filter gains fields as Yellowstone
            // gains features, and naming them all here means a dependency
            // bump breaks the build for no reason.
            ..Default::default()
        },
    );
    let request = pb::SubscribeRequest {
        transactions: filters,
        commitment: Some(pb::CommitmentLevel::Processed as i32),
        ..Default::default()
    };

    let outbound = futures_util::stream::iter(vec![request]);
    let response = client
        .subscribe(outbound)
        .await
        .map_err(|e| format!("subscribe: {e}"))?;
    let mut inbound = response.into_inner();

    while sink.running.load(Ordering::Relaxed) {
        let message = match inbound.next().await {
            Some(Ok(message)) => message,
            Some(Err(status)) => return Err(format!("stream: {status}")),
            None => return Err("stream ended".to_string()),
        };
        // Stamped before any decoding of our own, so the number measures the
        // wire rather than this function.
        let received_ns = now_ns();
        sink.stats.updates.fetch_add(1, Ordering::Relaxed);
        let transaction = match message.update_oneof {
            Some(pb::subscribe_update::UpdateOneof::Transaction(value)) => value,
            _ => continue,
        };
        sink.stats.transactions.fetch_add(1, Ordering::Relaxed);
        let signature = transaction
            .transaction
            .as_ref()
            .map(|info| info.signature.clone())
            .unwrap_or_default();
        if !sink.first_sight(&signature) {
            sink.stats.duplicates.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        for event in extract(&transaction, programs, received_ns) {
            sink.stats.matched.fetch_add(1, Ordering::Relaxed);
            sink.push(event);
        }
    }
    Ok(())
}
