//! Pump program event decoding, in Rust.
//!
//! This is the per-event half of the T0 receive path. Every trade and every
//! creation the stream carries passes through here -- 23,000 trades in a few
//! minutes on a normal feed -- and until now each one was parsed in Python:
//! a `struct.unpack_from` per field, a bytes slice per pubkey, a dict per
//! event, all on the path where an event arrives and something has to be
//! done about it.
//!
//! What this module is NOT: the socket. Owning the gRPC subscription would
//! mean adding an async runtime and a protobuf stack to a crate whose only
//! dependencies are bs58 and sha2, and rewriting reconnect and backpressure
//! logic that currently works. Decoding is the part that is pure
//! computation, runs per event, and can be proved identical to the Python
//! it replaces. That proof is the point: a decoder that disagrees with the
//! one the desk was built on is worse than a slow decoder.
//!
//! Layouts are those of the deployed program, matched field for field
//! against `src/chains/yellowstone_grpc.py`:
//!
//! ```text
//! TradeEvent     8 discriminator | 32 mint | 8 sol | 8 token | 1 is_buy
//!                32 user | 8 timestamp | [8 virtual_sol] [8 virtual_token]
//! CreateEvent    8 discriminator | borsh name, symbol, uri
//!                32 mint | 32 curve | 32 user | 32 creator | 8 timestamp
//! CompleteEvent  8 discriminator | 32 user | 32 mint | 32 curve | 8 timestamp
//! ```
//!
//! The two trailing TradeEvent fields are optional in the wire format and
//! are the ones that matter most: they carry the curve reserves, which is
//! what lets liquidity be priced at T0 without an RPC call. Absent, they
//! are None rather than zero -- an unknown reserve and an empty curve are
//! different facts.

use crate::helpers::b58encode;

/// `sha256("event:TradeEvent")[..8]`, as deployed.
pub const TRADE_EVENT: [u8; 8] = [189, 219, 127, 211, 78, 230, 97, 238];
pub const CREATE_EVENT: [u8; 8] = [27, 114, 169, 77, 222, 235, 99, 118];
pub const COMPLETE_EVENT: [u8; 8] = [95, 114, 97, 156, 212, 46, 152, 8];

/// Longest borsh string this will accept. A length prefix larger than this
/// is a malformed or hostile payload, not a very long token name.
const MAX_STRING: usize = 4_096;

#[derive(Debug, Clone, PartialEq)]
pub enum PumpEvent {
    Trade {
        mint: String,
        user: String,
        is_buy: bool,
        sol_amount: u64,
        token_amount: u64,
        timestamp: i64,
        /// None when the payload predates these fields. Distinguished from
        /// zero: an unknown reserve is not an empty curve.
        virtual_sol_reserves: Option<u64>,
        virtual_token_reserves: Option<u64>,
    },
    Create {
        mint: String,
        bonding_curve: String,
        user: String,
        creator: String,
        name: String,
        symbol: String,
        uri: String,
        timestamp: i64,
    },
    Complete {
        mint: String,
        user: String,
        bonding_curve: String,
        timestamp: i64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Not one of the three discriminators. Not an error in itself -- the
    /// stream carries plenty of other events -- so callers usually skip.
    UnknownDiscriminator,
    /// The payload ended before a field this layout requires.
    Truncated { event: &'static str, need: usize, got: usize },
    /// A borsh length prefix that cannot be honest.
    InvalidStringLength { offset: usize, length: usize },
}

fn u64_at(data: &[u8], offset: usize) -> u64 {
    let mut buffer = [0u8; 8];
    buffer.copy_from_slice(&data[offset..offset + 8]);
    u64::from_le_bytes(buffer)
}

fn i64_at(data: &[u8], offset: usize) -> i64 {
    u64_at(data, offset) as i64
}

fn pubkey_at(data: &[u8], offset: usize) -> String {
    b58encode(&data[offset..offset + 32])
}

/// Three borsh strings back to back, returning them and the offset after.
fn parse_create_strings(data: &[u8]) -> Result<(String, String, String, usize), DecodeError> {
    let mut values: Vec<String> = Vec::with_capacity(3);
    let mut offset = 0usize;
    for _ in 0..3 {
        if offset + 4 > data.len() {
            return Err(DecodeError::Truncated {
                event: "CreateEvent string",
                need: offset + 4,
                got: data.len(),
            });
        }
        let mut prefix = [0u8; 4];
        prefix.copy_from_slice(&data[offset..offset + 4]);
        let length = u32::from_le_bytes(prefix) as usize;
        offset += 4;
        if length > MAX_STRING || offset + length > data.len() {
            return Err(DecodeError::InvalidStringLength { offset, length });
        }
        // Lossy on purpose: a token name is attacker-controlled and may be
        // any bytes at all. Refusing to decode the whole event over an
        // invalid UTF-8 symbol would discard a launch for a cosmetic field.
        values.push(String::from_utf8_lossy(&data[offset..offset + length]).into_owned());
        offset += length;
    }
    let uri = values.pop().unwrap();
    let symbol = values.pop().unwrap();
    let name = values.pop().unwrap();
    Ok((name, symbol, uri, offset))
}

/// Decode one Pump program event from its CPI payload.
pub fn decode(data: &[u8]) -> Result<PumpEvent, DecodeError> {
    if data.len() < 8 {
        return Err(DecodeError::Truncated { event: "discriminator", need: 8, got: data.len() });
    }
    let discriminator = &data[..8];

    if discriminator == TRADE_EVENT {
        if data.len() < 97 {
            return Err(DecodeError::Truncated { event: "TradeEvent", need: 97, got: data.len() });
        }
        return Ok(PumpEvent::Trade {
            mint: pubkey_at(data, 8),
            sol_amount: u64_at(data, 40),
            token_amount: u64_at(data, 48),
            is_buy: data[56] != 0,
            user: pubkey_at(data, 57),
            timestamp: i64_at(data, 89),
            // Optional tail. Present on current payloads, absent on older
            // ones, and the difference decides whether T0 liquidity is
            // priceable without an RPC round trip.
            virtual_sol_reserves: (data.len() >= 105).then(|| u64_at(data, 97)),
            virtual_token_reserves: (data.len() >= 113).then(|| u64_at(data, 105)),
        });
    }

    if discriminator == CREATE_EVENT {
        let (name, symbol, uri, mut offset) = parse_create_strings(&data[8..])?;
        offset += 8;
        if data.len() < offset + 136 {
            return Err(DecodeError::Truncated {
                event: "CreateEvent",
                need: offset + 136,
                got: data.len(),
            });
        }
        return Ok(PumpEvent::Create {
            mint: pubkey_at(data, offset),
            bonding_curve: pubkey_at(data, offset + 32),
            user: pubkey_at(data, offset + 64),
            creator: pubkey_at(data, offset + 96),
            name,
            symbol,
            uri,
            timestamp: i64_at(data, offset + 128),
        });
    }

    if discriminator == COMPLETE_EVENT {
        if data.len() < 112 {
            return Err(DecodeError::Truncated {
                event: "CompleteEvent",
                need: 112,
                got: data.len(),
            });
        }
        return Ok(PumpEvent::Complete {
            user: pubkey_at(data, 8),
            mint: pubkey_at(data, 40),
            bonding_curve: pubkey_at(data, 72),
            timestamp: i64_at(data, 104),
        });
    }

    Err(DecodeError::UnknownDiscriminator)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trade_payload(with_reserves: bool) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&TRADE_EVENT);
        data.extend_from_slice(&[7u8; 32]); // mint
        data.extend_from_slice(&1_000u64.to_le_bytes());
        data.extend_from_slice(&2_000u64.to_le_bytes());
        data.push(1); // is_buy
        data.extend_from_slice(&[9u8; 32]); // user
        data.extend_from_slice(&1_700_000_000i64.to_le_bytes());
        if with_reserves {
            data.extend_from_slice(&30_000_000_000u64.to_le_bytes());
            data.extend_from_slice(&1_073_000_000_000_000u64.to_le_bytes());
        }
        data
    }

    #[test]
    fn a_trade_decodes_every_field() {
        match decode(&trade_payload(true)).unwrap() {
            PumpEvent::Trade { sol_amount, token_amount, is_buy, timestamp,
                               virtual_sol_reserves, virtual_token_reserves, .. } => {
                assert_eq!(sol_amount, 1_000);
                assert_eq!(token_amount, 2_000);
                assert!(is_buy);
                assert_eq!(timestamp, 1_700_000_000);
                assert_eq!(virtual_sol_reserves, Some(30_000_000_000));
                assert_eq!(virtual_token_reserves, Some(1_073_000_000_000_000));
            }
            other => panic!("expected a trade, got {other:?}"),
        }
    }

    #[test]
    fn absent_reserves_are_none_not_zero() {
        // An unknown reserve and an empty curve are different facts, and
        // only one of them means the token cannot be priced.
        match decode(&trade_payload(false)).unwrap() {
            PumpEvent::Trade { virtual_sol_reserves, virtual_token_reserves, .. } => {
                assert_eq!(virtual_sol_reserves, None);
                assert_eq!(virtual_token_reserves, None);
            }
            other => panic!("expected a trade, got {other:?}"),
        }
    }

    #[test]
    fn a_sell_is_not_read_as_a_buy() {
        let mut data = trade_payload(true);
        data[56] = 0;
        match decode(&data).unwrap() {
            PumpEvent::Trade { is_buy, .. } => assert!(!is_buy),
            other => panic!("expected a trade, got {other:?}"),
        }
    }

    #[test]
    fn a_truncated_trade_is_refused() {
        let data = trade_payload(true);
        assert!(matches!(
            decode(&data[..90]),
            Err(DecodeError::Truncated { event: "TradeEvent", .. })
        ));
    }

    #[test]
    fn a_creation_decodes_its_strings_and_keys() {
        let mut data = Vec::new();
        data.extend_from_slice(&CREATE_EVENT);
        for text in ["Dog", "DOG", "https://x/y"] {
            data.extend_from_slice(&(text.len() as u32).to_le_bytes());
            data.extend_from_slice(text.as_bytes());
        }
        for filler in [1u8, 2, 3, 4] {
            data.extend_from_slice(&[filler; 32]);
        }
        data.extend_from_slice(&1_700_000_001i64.to_le_bytes());
        match decode(&data).unwrap() {
            PumpEvent::Create { name, symbol, uri, timestamp, .. } => {
                assert_eq!(name, "Dog");
                assert_eq!(symbol, "DOG");
                assert_eq!(uri, "https://x/y");
                assert_eq!(timestamp, 1_700_000_001);
            }
            other => panic!("expected a creation, got {other:?}"),
        }
    }

    #[test]
    fn an_absurd_string_length_is_refused() {
        let mut data = Vec::new();
        data.extend_from_slice(&CREATE_EVENT);
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            decode(&data),
            Err(DecodeError::InvalidStringLength { .. })
        ));
    }

    #[test]
    fn invalid_utf8_in_a_name_does_not_lose_the_launch() {
        let mut data = Vec::new();
        data.extend_from_slice(&CREATE_EVENT);
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&[0xff, 0xfe]);       // not UTF-8
        for text in ["S", "u"] {
            data.extend_from_slice(&(text.len() as u32).to_le_bytes());
            data.extend_from_slice(text.as_bytes());
        }
        for filler in [1u8, 2, 3, 4] {
            data.extend_from_slice(&[filler; 32]);
        }
        data.extend_from_slice(&5i64.to_le_bytes());
        assert!(matches!(decode(&data), Ok(PumpEvent::Create { .. })));
    }

    #[test]
    fn an_unknown_discriminator_is_reported_as_such() {
        assert_eq!(decode(&[0u8; 32]).unwrap_err(), DecodeError::UnknownDiscriminator);
    }
}
