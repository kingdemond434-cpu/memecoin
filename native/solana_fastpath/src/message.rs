//! Compiling and serialising a v0 transaction message.
//!
//! This is the step between "we know which accounts and what data" and "we
//! have bytes to sign", and it is the step where a mistake is silent. A
//! message whose account order is wrong, or whose header counts are off by
//! one, serialises perfectly, signs perfectly, and is rejected by the runtime
//! with an error that points at the wrong thing. So the ordering rules are
//! stated here rather than assumed:
//!
//! Accounts are deduplicated and sorted into exactly four groups, in this
//! order, and the header counts describe the boundaries:
//!
//!   1. signers that are writable      (the fee payer is always first)
//!   2. signers that are read-only
//!   3. non-signers that are writable
//!   4. non-signers that are read-only  (program ids land here)
//!
//! Three rules make deduplication correct rather than merely tidy. Privileges
//! ACCUMULATE: an account appearing twice, once read-only and once writable,
//! is writable, and demoting it produces a transaction the runtime refuses to
//! let write. The FEE PAYER is always index zero and always a writable signer
//! regardless of how it was passed, because the runtime debits it. And within
//! each group the keys are SORTED BY PUBKEY, not left in insertion order --
//! solana-sdk holds them in a `BTreeMap`, so the ordering is a property of
//! the key bytes rather than of the order a caller happened to list them.
//!
//! That last rule is the one worth stating loudly. A message that groups
//! correctly but orders within a group differently is still VALID: the
//! runtime does not care. It is simply not the same bytes every other builder
//! on the network produces, which means a signature over it is not comparable
//! to anything, and the difference is invisible until something diffs the two.
//! The parity test against solders is what makes that difference visible, and
//! it found exactly this bug the first time it ran.
//!
//! Lengths are short-vec (compact-u16), not fixed-width: seven bits per byte,
//! high bit as continuation. Writing a plain u16 there is the classic way to
//! produce bytes that decode as something else entirely.
//!
//! v0 rather than legacy: the leading byte is `0x80`, the version tag with the
//! high bit set. Address table lookups are supported in the format and
//! deliberately emitted as an empty list here -- a lookup table saves bytes on
//! a large transaction and costs an extra account read, and on a 27-account
//! entry we would rather pay the bytes than the read.

use crate::instruction::{AccountMeta, Instruction, Pubkey, PUBKEY_LEN};

/// The v0 prefix: high bit set marks a versioned message, low bits the version.
pub const MESSAGE_VERSION_PREFIX: u8 = 0x80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageError {
    /// More than 255 distinct accounts, or more than 255 instructions. Both
    /// are runtime limits and both are caller bugs.
    TooManyAccounts,
    TooManyInstructions,
    /// An instruction referenced an account that is not in the compiled list.
    /// Impossible by construction here, and checked anyway because the cost
    /// of being wrong is a rejected transaction with a misleading error.
    UnknownAccount,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledMessage {
    pub num_required_signatures: u8,
    pub num_readonly_signed: u8,
    pub num_readonly_unsigned: u8,
    pub account_keys: Vec<Pubkey>,
    pub recent_blockhash: Pubkey,
    pub instructions: Vec<CompiledInstruction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledInstruction {
    pub program_id_index: u8,
    pub account_indexes: Vec<u8>,
    pub data: Vec<u8>,
}

/// Short-vec: seven bits of length per byte, high bit continues.
pub fn encode_length(value: usize, out: &mut Vec<u8>) {
    let mut remaining = value;
    loop {
        let mut byte = (remaining & 0x7f) as u8;
        remaining >>= 7;
        if remaining == 0 {
            out.push(byte);
            return;
        }
        byte |= 0x80;
        out.push(byte);
    }
}

#[derive(Clone, Copy)]
struct Slot {
    key: Pubkey,
    is_signer: bool,
    is_writable: bool,
}

/// Compile instructions plus a fee payer into a v0 message.
pub fn compile(payer: &Pubkey, instructions: &[Instruction], recent_blockhash: &Pubkey)
    -> Result<CompiledMessage, MessageError>
{
    if instructions.len() > u8::MAX as usize {
        return Err(MessageError::TooManyInstructions);
    }
    // The payer is seeded first so it holds index zero even if it also appears
    // inside an instruction as a read-only account.
    let mut slots: Vec<Slot> = vec![Slot { key: *payer, is_signer: true, is_writable: true }];

    let absorb = |meta: &AccountMeta, slots: &mut Vec<Slot>| {
        if let Some(existing) = slots.iter_mut().find(|slot| slot.key == meta.pubkey) {
            // Privileges accumulate. Taking the last-seen value instead would
            // let a later read-only mention demote an account the transaction
            // needs to write.
            existing.is_signer |= meta.is_signer;
            existing.is_writable |= meta.is_writable;
        } else {
            slots.push(Slot { key: meta.pubkey, is_signer: meta.is_signer,
                              is_writable: meta.is_writable });
        }
    };

    // Program id FIRST, then that instruction's accounts. This ordering is
    // not cosmetic and it is not ours to choose: solana-sdk's CompiledKeys
    // absorbs `ix.program_id` before iterating `ix.accounts`, and within each
    // privilege group the account list is insertion-ordered. Absorbing
    // programs in a second pass produces a message that is still VALID --
    // the runtime does not care about order inside a group -- and is not
    // byte-identical to what every other builder on the network produces.
    // That difference is invisible until a parity check catches it, which is
    // the whole reason there is one.
    for instruction in instructions {
        absorb(&AccountMeta { pubkey: instruction.program_id,
                              is_signer: false, is_writable: false }, &mut slots);
        for meta in &instruction.accounts {
            absorb(meta, &mut slots);
        }
    }

    let mut writable_signers: Vec<Slot> = Vec::new();
    let mut readonly_signers: Vec<Slot> = Vec::new();
    let mut writable_others: Vec<Slot> = Vec::new();
    let mut readonly_others: Vec<Slot> = Vec::new();
    for slot in slots {
        if slot.key == *payer {
            // Forced to index zero below, so it never joins a sorted group.
            continue;
        }
        match (slot.is_signer, slot.is_writable) {
            (true, true) => writable_signers.push(slot),
            (true, false) => readonly_signers.push(slot),
            (false, true) => writable_others.push(slot),
            (false, false) => readonly_others.push(slot),
        }
    }
    // Sorted by key, matching solana-sdk's BTreeMap. Insertion order here
    // produces a valid message that is not byte-identical to anyone else's.
    for group in [&mut writable_signers, &mut readonly_signers,
                  &mut writable_others, &mut readonly_others] {
        group.sort_by(|left, right| left.key.cmp(&right.key));
    }
    writable_signers.insert(0, Slot { key: *payer, is_signer: true, is_writable: true });

    let num_required_signatures = writable_signers.len() + readonly_signers.len();
    let ordered: Vec<Slot> = writable_signers.iter()
        .chain(readonly_signers.iter())
        .chain(writable_others.iter())
        .chain(readonly_others.iter())
        .copied()
        .collect();
    if ordered.len() > u8::MAX as usize || num_required_signatures > u8::MAX as usize {
        return Err(MessageError::TooManyAccounts);
    }

    let account_keys: Vec<Pubkey> = ordered.iter().map(|slot| slot.key).collect();
    let index_of = |key: &Pubkey| -> Option<u8> {
        account_keys.iter().position(|candidate| candidate == key).map(|index| index as u8)
    };

    let mut compiled = Vec::with_capacity(instructions.len());
    for instruction in instructions {
        let program_id_index = index_of(&instruction.program_id)
            .ok_or(MessageError::UnknownAccount)?;
        let mut account_indexes = Vec::with_capacity(instruction.accounts.len());
        for meta in &instruction.accounts {
            account_indexes.push(index_of(&meta.pubkey).ok_or(MessageError::UnknownAccount)?);
        }
        compiled.push(CompiledInstruction {
            program_id_index,
            account_indexes,
            data: instruction.data.clone(),
        });
    }

    Ok(CompiledMessage {
        num_required_signatures: num_required_signatures as u8,
        num_readonly_signed: readonly_signers.len() as u8,
        num_readonly_unsigned: readonly_others.len() as u8,
        account_keys,
        recent_blockhash: *recent_blockhash,
        instructions: compiled,
    })
}

impl CompiledMessage {
    /// The exact bytes that get signed, version prefix included.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            1 + 3 + 1 + self.account_keys.len() * PUBKEY_LEN + PUBKEY_LEN + 64);
        out.push(MESSAGE_VERSION_PREFIX);
        out.push(self.num_required_signatures);
        out.push(self.num_readonly_signed);
        out.push(self.num_readonly_unsigned);
        encode_length(self.account_keys.len(), &mut out);
        for key in &self.account_keys {
            out.extend_from_slice(key);
        }
        out.extend_from_slice(&self.recent_blockhash);
        encode_length(self.instructions.len(), &mut out);
        for instruction in &self.instructions {
            out.push(instruction.program_id_index);
            encode_length(instruction.account_indexes.len(), &mut out);
            out.extend_from_slice(&instruction.account_indexes);
            encode_length(instruction.data.len(), &mut out);
            out.extend_from_slice(&instruction.data);
        }
        // Address table lookups: an empty short-vec. Present in the format
        // and deliberately unused -- see the module note.
        encode_length(0, &mut out);
        out
    }

    pub fn signers(&self) -> usize {
        self.num_required_signatures as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(byte: u8) -> Pubkey {
        [byte; PUBKEY_LEN]
    }

    fn instruction(program: u8, metas: Vec<AccountMeta>, data: Vec<u8>) -> Instruction {
        Instruction { program_id: key(program), accounts: metas, data }
    }

    #[test]
    fn the_fee_payer_is_index_zero_and_writable_even_when_passed_read_only() {
        let payer = key(1);
        let message = compile(&payer, &[instruction(9, vec![
            AccountMeta { pubkey: payer, is_signer: false, is_writable: false },
        ], vec![])], &key(0)).unwrap();
        assert_eq!(message.account_keys[0], payer);
        assert_eq!(message.num_required_signatures, 1);
        // Writable signer, so it is not counted among the read-only signers.
        assert_eq!(message.num_readonly_signed, 0);
    }

    #[test]
    fn privileges_accumulate_across_duplicate_mentions() {
        let payer = key(1);
        let shared = key(2);
        let message = compile(&payer, &[
            instruction(9, vec![
                AccountMeta { pubkey: shared, is_signer: false, is_writable: false }], vec![]),
            instruction(9, vec![
                AccountMeta { pubkey: shared, is_signer: false, is_writable: true }], vec![]),
        ], &key(0)).unwrap();
        // Writable wins, so `shared` sorts before the program id even though
        // the program was absorbed first.
        let shared_index = message.account_keys.iter().position(|k| *k == shared).unwrap();
        let program_index = message.account_keys.iter().position(|k| *k == key(9)).unwrap();
        assert!(shared_index < program_index);
        assert_eq!(message.num_readonly_unsigned, 1);
    }

    #[test]
    fn accounts_are_grouped_signers_then_writables_then_readonly() {
        let message = compile(&key(1), &[instruction(9, vec![
            AccountMeta { pubkey: key(3), is_signer: true, is_writable: false },
            AccountMeta { pubkey: key(4), is_signer: false, is_writable: true },
            AccountMeta { pubkey: key(5), is_signer: false, is_writable: false },
        ], vec![])], &key(0)).unwrap();
        // Groups are sorted by key, and key(5) < key(9), so the program id
        // follows the read-only account rather than preceding it.
        assert_eq!(message.account_keys, vec![key(1), key(3), key(4), key(5), key(9)]);
        assert_eq!(message.num_required_signatures, 2);
        assert_eq!(message.num_readonly_signed, 1);
        assert_eq!(message.num_readonly_unsigned, 2);
    }


    #[test]
    fn each_privilege_group_is_sorted_by_key() {
        // Listed deliberately out of order: the compiled message must not
        // depend on the order a caller happened to pass them.
        let message = compile(&key(1), &[instruction(200, vec![
            AccountMeta { pubkey: key(80), is_signer: false, is_writable: true },
            AccountMeta { pubkey: key(20), is_signer: false, is_writable: true },
            AccountMeta { pubkey: key(90), is_signer: false, is_writable: false },
            AccountMeta { pubkey: key(30), is_signer: false, is_writable: false },
            AccountMeta { pubkey: key(70), is_signer: true, is_writable: false },
            AccountMeta { pubkey: key(40), is_signer: true, is_writable: false },
        ], vec![])], &key(0)).unwrap();
        assert_eq!(message.account_keys, vec![
            key(1),                     // payer, always index zero
            key(40), key(70),           // read-only signers, sorted
            key(20), key(80),           // writable non-signers, sorted
            key(30), key(90), key(200), // read-only non-signers, sorted
        ]);
    }

    #[test]
    fn the_payer_holds_index_zero_regardless_of_its_key_bytes() {
        // A payer whose bytes sort last would drift out of index zero under
        // a plain sort, and the runtime debits index zero.
        let message = compile(&key(255), &[instruction(9, vec![
            AccountMeta { pubkey: key(2), is_signer: true, is_writable: true },
        ], vec![])], &key(0)).unwrap();
        assert_eq!(message.account_keys[0], key(255));
        assert_eq!(message.account_keys[1], key(2));
    }

    #[test]
    fn serialised_bytes_start_with_the_v0_prefix() {
        let message = compile(&key(1), &[instruction(9, vec![], vec![7])], &key(0)).unwrap();
        assert_eq!(message.serialize()[0], MESSAGE_VERSION_PREFIX);
    }

    #[test]
    fn short_vec_uses_seven_bits_per_byte() {
        let mut out = Vec::new();
        encode_length(0, &mut out);
        assert_eq!(out, vec![0]);
        out.clear();
        encode_length(127, &mut out);
        assert_eq!(out, vec![127]);
        out.clear();
        encode_length(128, &mut out);
        assert_eq!(out, vec![0x80, 0x01]);
        out.clear();
        encode_length(16384, &mut out);
        assert_eq!(out, vec![0x80, 0x80, 0x01]);
    }

    #[test]
    fn a_lookup_table_list_is_present_and_empty() {
        let message = compile(&key(1), &[instruction(9, vec![], vec![])], &key(0)).unwrap();
        assert_eq!(*message.serialize().last().unwrap(), 0u8);
    }
}
