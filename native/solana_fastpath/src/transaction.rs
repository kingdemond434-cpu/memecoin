//! Signing and assembling the transaction the wire actually carries.
//!
//! The last two steps of the entry path, and the two that have never been in
//! Rust: sign the compiled message, then prepend the signatures and encode.
//! Signing an Ed25519 message costs tens of microseconds here; the win over
//! Python is not the signature itself but everything around it -- no object
//! allocation per account, no re-serialisation, no round trip through a
//! library that rebuilds the message to sign it.
//!
//! Three things this module is deliberate about.
//!
//! **The secret never leaves the caller's buffer for longer than the call.**
//! A signing key is taken as bytes, used, and dropped; `zeroize` clears it on
//! the way out. Nothing here logs, formats, or returns a key, and there is no
//! code path that could -- the only thing that comes back is a signature and
//! an encoded transaction.
//!
//! **A 64-byte "keypair" is the Solana convention and it is checked, not
//! trusted.** Solana wallets store secret-then-public. If a caller passes
//! those 64 bytes, the public half is VERIFIED against the one derived from
//! the secret rather than being taken on faith: a mismatched pair produces a
//! signature that is valid for a different account, which the runtime rejects
//! after the slot is gone.
//!
//! **Unsigned slots are zero-filled rather than omitted.** A transaction
//! requiring two signatures and carrying one is malformed; carrying one real
//! signature and one zeroed placeholder is the documented shape for a
//! partially signed transaction, which is what a co-signed or fee-sponsored
//! flow needs.

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use ed25519_dalek::{Signer, SigningKey};
use zeroize::Zeroize;

use crate::instruction::{Instruction, Pubkey, PUBKEY_LEN};
use crate::message::{compile, encode_length, CompiledMessage, MessageError};

pub const SIGNATURE_LEN: usize = 64;
pub const SECRET_LEN: usize = 32;
pub const KEYPAIR_LEN: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignError {
    /// Not 32 (secret only) or 64 (secret then public) bytes.
    BadKeyLength(usize),
    /// The public half of a 64-byte keypair does not match the secret half.
    /// Signing anyway would produce a valid signature for the wrong account.
    KeypairMismatch,
    /// The message needs a signature from a key the caller did not supply.
    MissingSigner(usize),
    Message(MessageError),
}

/// A signing key plus the account it signs for.
pub struct Signer32 {
    key: SigningKey,
    pub public: Pubkey,
}

impl Signer32 {
    /// Build from 32 secret bytes, or 64 secret-then-public bytes.
    ///
    /// The input is copied and the copy is zeroized before returning, so a
    /// caller passing a slice into a Python buffer does not leave a second
    /// live copy of the secret in this process.
    pub fn from_bytes(raw: &[u8]) -> Result<Self, SignError> {
        let mut secret = [0u8; SECRET_LEN];
        match raw.len() {
            SECRET_LEN => secret.copy_from_slice(raw),
            KEYPAIR_LEN => secret.copy_from_slice(&raw[..SECRET_LEN]),
            other => return Err(SignError::BadKeyLength(other)),
        }
        let key = SigningKey::from_bytes(&secret);
        secret.zeroize();
        let public: Pubkey = key.verifying_key().to_bytes();
        if raw.len() == KEYPAIR_LEN && raw[SECRET_LEN..] != public {
            return Err(SignError::KeypairMismatch);
        }
        Ok(Self { key, public })
    }

    pub fn sign(&self, message: &[u8]) -> [u8; SIGNATURE_LEN] {
        self.key.sign(message).to_bytes()
    }
}

#[derive(Debug)]
pub struct SignedTransaction {
    pub signatures: Vec<[u8; SIGNATURE_LEN]>,
    pub message: CompiledMessage,
    pub serialized_message: Vec<u8>,
}

impl SignedTransaction {
    /// Wire bytes: short-vec of signatures, then the message.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            2 + self.signatures.len() * SIGNATURE_LEN + self.serialized_message.len());
        encode_length(self.signatures.len(), &mut out);
        for signature in &self.signatures {
            out.extend_from_slice(signature);
        }
        out.extend_from_slice(&self.serialized_message);
        out
    }

    pub fn to_base64(&self) -> String {
        BASE64.encode(self.serialize())
    }

    /// The transaction's identity on chain: its first signature, base58.
    pub fn signature_b58(&self) -> String {
        match self.signatures.first() {
            Some(signature) => crate::helpers::b58encode(signature),
            None => String::new(),
        }
    }
}

/// Compile, sign and assemble in one pass.
///
/// Every account the message requires a signature from must be covered by one
/// of `signers`, EXCEPT that a signer the caller genuinely cannot supply is
/// reported by index rather than silently zero-filled. Silent zero-filling is
/// how an unsigned transaction reaches the wire and comes back as a generic
/// failure a slot later.
pub fn build_signed(payer: &Pubkey, instructions: &[Instruction],
                    recent_blockhash: &Pubkey, signers: &[Signer32],
                    allow_partial: bool)
    -> Result<SignedTransaction, SignError>
{
    let message = compile(payer, instructions, recent_blockhash)
        .map_err(SignError::Message)?;
    let serialized_message = message.serialize();
    let required = message.signers();
    let mut signatures = Vec::with_capacity(required);
    for index in 0..required {
        let account = message.account_keys[index];
        match signers.iter().find(|signer| signer.public == account) {
            Some(signer) => signatures.push(signer.sign(&serialized_message)),
            None if allow_partial => signatures.push([0u8; SIGNATURE_LEN]),
            None => return Err(SignError::MissingSigner(index)),
        }
    }
    Ok(SignedTransaction { signatures, message, serialized_message })
}

/// Sign a message somebody else compiled.
///
/// The bridge that makes incremental adoption safe: the Python path can build
/// a message with solders, hand the bytes here, and compare. Byte-identical
/// output is the only evidence that promoting this path is safe with real
/// capital on it.
pub fn sign_serialized_message(serialized_message: &[u8], signers: &[Signer32])
    -> Vec<[u8; SIGNATURE_LEN]>
{
    signers.iter().map(|signer| signer.sign(serialized_message)).collect()
}

/// Assemble a transaction from a message somebody else signed.
///
/// The half of the path that matters for a desk whose signing key lives in a
/// separate process. `build_signed` needs the secret; this needs only the
/// signatures, so the whole isolation architecture survives being moved to
/// Rust. Compile here, hand the message bytes to the signer, assemble the
/// answer here. The key never enters this process.
///
/// Signatures are taken in the message's own signer order and are NOT
/// reordered or matched by pubkey -- this function cannot see the public keys
/// to match them against. A caller that supplies them out of order produces a
/// transaction that fails, which is why the only caller is the one that
/// compiled the message.
pub fn assemble(serialized_message: &[u8], signatures: &[Vec<u8>])
    -> Result<String, SignError>
{
    let mut out = Vec::with_capacity(
        2 + signatures.len() * SIGNATURE_LEN + serialized_message.len());
    encode_length(signatures.len(), &mut out);
    for signature in signatures {
        if signature.len() != SIGNATURE_LEN {
            return Err(SignError::BadKeyLength(signature.len()));
        }
        out.extend_from_slice(signature);
    }
    out.extend_from_slice(serialized_message);
    Ok(BASE64.encode(out))
}

pub fn pubkey_of(raw: &[u8]) -> Option<Pubkey> {
    if raw.len() != PUBKEY_LEN {
        return None;
    }
    let mut out = [0u8; PUBKEY_LEN];
    out.copy_from_slice(raw);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction::AccountMeta;
    use ed25519_dalek::{Verifier, VerifyingKey};

    fn signer(seed: u8) -> Signer32 {
        Signer32::from_bytes(&[seed; SECRET_LEN]).expect("valid secret")
    }

    fn instruction(program: Pubkey, metas: Vec<AccountMeta>) -> Instruction {
        Instruction { program_id: program, accounts: metas, data: vec![1, 2, 3] }
    }

    #[test]
    fn a_signed_transaction_verifies_against_the_message_it_carries() {
        let payer = signer(7);
        let transaction = build_signed(
            &payer.public,
            &[instruction([9u8; PUBKEY_LEN], vec![])],
            &[3u8; PUBKEY_LEN], &[signer(7)], false).expect("signed");
        let verifying = VerifyingKey::from_bytes(&payer.public).unwrap();
        let signature = ed25519_dalek::Signature::from_bytes(&transaction.signatures[0]);
        assert!(verifying.verify(&transaction.serialized_message, &signature).is_ok());
    }

    #[test]
    fn a_missing_signer_is_named_rather_than_zero_filled() {
        let payer = signer(7);
        let error = build_signed(&payer.public, &[instruction([9u8; PUBKEY_LEN], vec![])],
                                 &[3u8; PUBKEY_LEN], &[], false).unwrap_err();
        assert_eq!(error, SignError::MissingSigner(0));
    }

    #[test]
    fn partial_signing_zero_fills_only_when_asked(  ) {
        let payer = signer(7);
        let transaction = build_signed(&payer.public, &[instruction([9u8; PUBKEY_LEN], vec![])],
                                       &[3u8; PUBKEY_LEN], &[], true).expect("partial");
        assert_eq!(transaction.signatures[0], [0u8; SIGNATURE_LEN]);
    }

    #[test]
    fn a_mismatched_keypair_is_refused_before_it_signs_for_the_wrong_account() {
        let mut raw = [0u8; KEYPAIR_LEN];
        raw[..SECRET_LEN].copy_from_slice(&[7u8; SECRET_LEN]);
        raw[SECRET_LEN..].copy_from_slice(&[9u8; SECRET_LEN]);
        assert_eq!(Signer32::from_bytes(&raw).err(), Some(SignError::KeypairMismatch));
    }

    #[test]
    fn a_wallet_format_keypair_is_accepted() {
        let key = signer(11);
        let mut raw = [0u8; KEYPAIR_LEN];
        raw[..SECRET_LEN].copy_from_slice(&[11u8; SECRET_LEN]);
        raw[SECRET_LEN..].copy_from_slice(&key.public);
        assert_eq!(Signer32::from_bytes(&raw).unwrap().public, key.public);
    }

    #[test]
    fn a_short_key_is_refused_rather_than_padded() {
        assert_eq!(Signer32::from_bytes(&[1, 2, 3]).err(), Some(SignError::BadKeyLength(3)));
    }

    #[test]
    fn assembling_from_signatures_matches_signing_in_place() {
        // The two paths must be interchangeable, because one of them is what
        // an isolated signer forces and the other is what the tests exercise.
        let payer = signer(7);
        let instructions = [instruction([9u8; PUBKEY_LEN], vec![])];
        let built = build_signed(&payer.public, &instructions, &[3u8; PUBKEY_LEN],
                                 &[signer(7)], false).unwrap();
        let assembled = assemble(&built.serialized_message,
                                 &[built.signatures[0].to_vec()]).unwrap();
        assert_eq!(assembled, built.to_base64());
    }

    #[test]
    fn a_wrong_length_signature_is_refused_rather_than_padded() {
        assert!(assemble(&[0u8; 32], &[vec![1, 2, 3]]).is_err());
    }

    #[test]
    fn the_wire_form_starts_with_the_signature_count() {
        let payer = signer(7);
        let transaction = build_signed(&payer.public, &[instruction([9u8; PUBKEY_LEN], vec![])],
                                       &[3u8; PUBKEY_LEN], &[signer(7)], false).unwrap();
        let wire = transaction.serialize();
        assert_eq!(wire[0], 1);
        assert_eq!(&wire[1..1 + SIGNATURE_LEN], &transaction.signatures[0]);
        assert_eq!(wire[1 + SIGNATURE_LEN], super::super::message::MESSAGE_VERSION_PREFIX);
    }
}
