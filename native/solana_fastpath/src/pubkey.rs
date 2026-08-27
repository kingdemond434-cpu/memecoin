//! Program-derived addresses, computed here rather than after the event.
//!
//! A Pump `buy_v2` needs twenty-seven accounts and eleven of them are PDAs:
//! the bonding curve, the creator vault, four associated token accounts, the
//! volume accumulators, the fee config. Deriving one is a SHA-256 over the
//! seeds plus an off-curve check, and the off-curve check fails for roughly
//! one bump in every two hundred and fifty-six, so the average derivation is
//! about two hashes and the worst case is up to two hundred and fifty-five.
//!
//! Doing that eleven times in Python after a launch event arrives is real,
//! avoidable milliseconds on the one path where milliseconds are the product.
//! Doing it here is tens of microseconds. More importantly it can be done
//! BEFORE the decision: every seed except the trade size is known the instant
//! `token_created` is decoded, so the whole account list can be derived while
//! the policy is still thinking.
//!
//! The off-curve test is the part that is easy to get subtly wrong. Solana
//! requires a PDA to have no corresponding private key, which means the
//! 32-byte hash must NOT decompress to a valid Edwards point. `decompress()`
//! returning `Some` means the address IS on the curve and the bump must be
//! decremented. Skipping that check produces addresses that look right,
//! derive deterministically, and are rejected by the runtime -- a failure mode
//! that costs a slot and points nowhere.

use curve25519_dalek::edwards::CompressedEdwardsY;
use sha2::{Digest, Sha256};

use crate::instruction::{Pubkey, PUBKEY_LEN};

/// Appended to every PDA preimage so a derived address can never collide with
/// a plain hash of the same seeds.
const PDA_MARKER: &[u8] = b"ProgramDerivedAddress";

pub const MAX_SEEDS: usize = 16;
pub const MAX_SEED_LEN: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdaError {
    /// More than sixteen seeds, or one longer than thirty-two bytes. The
    /// runtime enforces both and a violation is a caller bug, not a retry.
    InvalidSeeds,
    /// All 256 bumps landed on the curve. Cryptographically negligible, and
    /// reported rather than unwrapped because a panic here would take down
    /// the decision loop.
    NoViableBump,
}

/// True when the bytes decompress to a valid Edwards point -- i.e. an address
/// that could have a private key, and therefore cannot be a PDA.
fn is_on_curve(bytes: &Pubkey) -> bool {
    CompressedEdwardsY(*bytes).decompress().is_some()
}

/// `create_program_address`: the address for one specific bump, or None when
/// that bump lands on the curve.
pub fn create_program_address(seeds: &[&[u8]], program_id: &Pubkey) -> Option<Pubkey> {
    if seeds.len() > MAX_SEEDS {
        return None;
    }
    let mut hasher = Sha256::new();
    for seed in seeds {
        if seed.len() > MAX_SEED_LEN {
            return None;
        }
        hasher.update(seed);
    }
    hasher.update(program_id);
    hasher.update(PDA_MARKER);
    let digest = hasher.finalize();
    let mut out = [0u8; PUBKEY_LEN];
    out.copy_from_slice(&digest);
    if is_on_curve(&out) {
        None
    } else {
        Some(out)
    }
}

/// `find_program_address`: the canonical address and its bump.
///
/// Descends from 255 exactly as the runtime does. Ascending would find a
/// different, equally valid-looking address that the program does not use,
/// and nothing downstream would report the difference until the transaction
/// failed against an account that does not exist.
pub fn find_program_address(seeds: &[&[u8]], program_id: &Pubkey)
    -> Result<(Pubkey, u8), PdaError>
{
    if seeds.len() >= MAX_SEEDS {
        return Err(PdaError::InvalidSeeds);
    }
    for seed in seeds {
        if seed.len() > MAX_SEED_LEN {
            return Err(PdaError::InvalidSeeds);
        }
    }
    let mut bump = 255u8;
    loop {
        let bump_seed = [bump];
        let mut with_bump: Vec<&[u8]> = Vec::with_capacity(seeds.len() + 1);
        with_bump.extend_from_slice(seeds);
        with_bump.push(&bump_seed);
        if let Some(address) = create_program_address(&with_bump, program_id) {
            return Ok((address, bump));
        }
        if bump == 0 {
            return Err(PdaError::NoViableBump);
        }
        bump -= 1;
    }
}

/// The SPL associated token account for (owner, mint) under a token program.
///
/// Takes the token program explicitly because Pump trades both SPL Token and
/// Token-2022 mints, and an ATA derived under the wrong program is a valid
/// address for an account that will never exist.
pub fn associated_token_address(owner: &Pubkey, token_program: &Pubkey, mint: &Pubkey,
                                associated_token_program: &Pubkey)
    -> Result<Pubkey, PdaError>
{
    find_program_address(&[owner, token_program, mint], associated_token_program)
        .map(|(address, _bump)| address)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{b58decode, b58encode};

    fn key(value: &str) -> Pubkey {
        let raw = b58decode(value).expect("valid base58");
        let mut out = [0u8; PUBKEY_LEN];
        out.copy_from_slice(&raw);
        out
    }

    #[test]
    fn associated_token_address_matches_the_spl_reference() {
        // Wrapped SOL's ATA for a known owner, cross-checked against
        // spl-associated-token-account. If this drifts, every quote account
        // on the entry path is wrong and nothing else here can be trusted.
        let owner = key("11111111111111111111111111111111");
        let mint = key("So11111111111111111111111111111111111111112");
        let token_program = key("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA");
        let ata_program = key("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL");
        let derived = associated_token_address(&owner, &token_program, &mint, &ata_program)
            .expect("derivable");
        // Expected value produced by solders (the Rust `solana-program`
        // implementation behind the Python path), not written from memory.
        // tests/test_core.py asserts the same address from the Python side,
        // so a drift in either implementation fails on both.
        assert_eq!(b58encode(&derived), "aqxoAhCwpy3oB1BpNw9hL1HdLYLgPpbPjzxDrrQj3Fs");
    }

    #[test]
    fn a_pda_is_never_a_point_on_the_curve() {
        let program = key("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P");
        let (address, bump) =
            find_program_address(&[b"bonding-curve"], &program).expect("derivable");
        assert!(!is_on_curve(&address));
        assert_eq!(b58encode(&address), "7cogN2h8NCWHcfPdWvpo5FMg61jefi8LLwHop3fk3p9y");
        assert_eq!(bump, 255);
    }

    #[test]
    fn derivation_is_deterministic() {
        let program = key("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P");
        let first = find_program_address(&[b"global"], &program).unwrap();
        let second = find_program_address(&[b"global"], &program).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn an_oversized_seed_is_refused_rather_than_truncated() {
        let program = key("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P");
        let long = [7u8; MAX_SEED_LEN + 1];
        assert_eq!(find_program_address(&[&long], &program), Err(PdaError::InvalidSeeds));
    }
}
