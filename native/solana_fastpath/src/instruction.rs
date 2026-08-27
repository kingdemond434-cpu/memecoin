//! Native Pump `buy_v2` / `sell_v2` instruction construction.
//!
//! Construction is where the CPU actually goes on the entry path: the account
//! list is 26-27 entries, several of them PDAs that have to be derived, and
//! doing that in Python after an event arrives is most of the avoidable
//! latency. Signing costs tens of microseconds and submission is
//! network-bound, so building here and signing on the existing gated path
//! captures nearly all of the win while leaving the live-capital lock exactly
//! where it is.
//!
//! The account order and every flag below match idl/pump.json, which is what
//! the program was compiled against. They were originally transcribed from the
//! prose tables in docs/instructions/BUY.md and SELL.md, and on three flags
//! those tables disagree with the program: `fee_recipient` and
//! `buyback_fee_recipient` are writable and the tables say they are not, and
//! `global_volume_accumulator` is NOT writable and the tables say it is. A
//! transaction that declares the wrong mutability fails without pointing at
//! why. The Python side generates its lists from the IDL directly (see
//! src/chains/idl.py); this side is checked against it by the parity test.
//!
//! Expressed as a fixed-length array rather than a builder: an account list a
//! caller can assemble in the wrong order is a transaction that fails, or
//! worse, succeeds against the wrong account.
//!
//! The two lists differ in more than length. `buy_v2` takes 27 accounts
//! including `global_volume_accumulator`; `sell_v2` takes 26 and omits it.
//! `user` is writable AND signer on both. Copying one list to the other is the
//! obvious mistake and it is one the type system here prevents.

use sha2::{Digest, Sha256};

pub const PUBKEY_LEN: usize = 32;
pub type Pubkey = [u8; PUBKEY_LEN];

pub const BUY_V2_ACCOUNTS: usize = 27;
pub const SELL_V2_ACCOUNTS: usize = 26;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccountMeta {
    pub pubkey: Pubkey,
    pub is_signer: bool,
    pub is_writable: bool,
}

impl AccountMeta {
    fn ro(pubkey: Pubkey) -> Self {
        Self { pubkey, is_signer: false, is_writable: false }
    }
    fn rw(pubkey: Pubkey) -> Self {
        Self { pubkey, is_signer: false, is_writable: true }
    }
    fn signer(pubkey: Pubkey, is_writable: bool) -> Self {
        Self { pubkey, is_signer: true, is_writable }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    pub program_id: Pubkey,
    pub accounts: Vec<AccountMeta>,
    pub data: Vec<u8>,
}

/// Anchor's instruction discriminator: `sha256("global:<name>")[..8]`.
pub fn anchor_instruction_discriminator(name: &str) -> [u8; 8] {
    let digest = Sha256::digest(format!("global:{name}").as_bytes());
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest[..8]);
    out
}

/// Anchor's account discriminator: `sha256("account:<Name>")[..8]`.
pub fn anchor_account_discriminator(name: &str) -> [u8; 8] {
    let digest = Sha256::digest(format!("account:{name}").as_bytes());
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest[..8]);
    out
}

/// Every account `buy_v2` needs, in the documented order.
#[derive(Debug, Clone, Copy)]
pub struct BuyAccounts {
    pub global: Pubkey,
    pub base_mint: Pubkey,
    pub quote_mint: Pubkey,
    pub base_token_program: Pubkey,
    pub quote_token_program: Pubkey,
    pub associated_token_program: Pubkey,
    pub fee_recipient: Pubkey,
    pub associated_quote_fee_recipient: Pubkey,
    pub buyback_fee_recipient: Pubkey,
    pub associated_quote_buyback_fee_recipient: Pubkey,
    pub bonding_curve: Pubkey,
    pub associated_base_bonding_curve: Pubkey,
    pub associated_quote_bonding_curve: Pubkey,
    pub user: Pubkey,
    pub associated_base_user: Pubkey,
    pub associated_quote_user: Pubkey,
    pub creator_vault: Pubkey,
    pub associated_creator_vault: Pubkey,
    pub sharing_config: Pubkey,
    pub global_volume_accumulator: Pubkey,
    pub user_volume_accumulator: Pubkey,
    pub associated_user_volume_accumulator: Pubkey,
    pub fee_config: Pubkey,
    pub fee_program: Pubkey,
    pub system_program: Pubkey,
    pub event_authority: Pubkey,
    pub program: Pubkey,
}

/// Every account `sell_v2` needs, in the documented order.
#[derive(Debug, Clone, Copy)]
pub struct SellAccounts {
    pub global: Pubkey,
    pub base_mint: Pubkey,
    pub quote_mint: Pubkey,
    pub base_token_program: Pubkey,
    pub quote_token_program: Pubkey,
    pub associated_token_program: Pubkey,
    pub fee_recipient: Pubkey,
    pub associated_quote_fee_recipient: Pubkey,
    pub buyback_fee_recipient: Pubkey,
    pub associated_quote_buyback_fee_recipient: Pubkey,
    pub bonding_curve: Pubkey,
    pub associated_base_bonding_curve: Pubkey,
    pub associated_quote_bonding_curve: Pubkey,
    pub user: Pubkey,
    pub associated_base_user: Pubkey,
    pub associated_quote_user: Pubkey,
    pub creator_vault: Pubkey,
    pub associated_creator_vault: Pubkey,
    pub sharing_config: Pubkey,
    pub user_volume_accumulator: Pubkey,
    pub associated_user_volume_accumulator: Pubkey,
    pub fee_config: Pubkey,
    pub fee_program: Pubkey,
    pub system_program: Pubkey,
    pub event_authority: Pubkey,
    pub program: Pubkey,
}

fn encode_args(discriminator: [u8; 8], first: u64, second: u64) -> Vec<u8> {
    let mut data = Vec::with_capacity(24);
    data.extend_from_slice(&discriminator);
    data.extend_from_slice(&first.to_le_bytes());
    data.extend_from_slice(&second.to_le_bytes());
    data
}

/// `buy_v2(amount, max_sol_cost)`.
///
/// `max_sol_cost` is the caller's slippage protection and is not derived
/// here. A construction layer that computed its own bound would be choosing
/// the trade's risk limit, which belongs to the sizing decision.
pub fn build_buy_v2(accounts: &BuyAccounts, amount: u64, max_sol_cost: u64) -> Instruction {
    let metas = vec![
        AccountMeta::ro(accounts.global),
        AccountMeta::ro(accounts.base_mint),
        AccountMeta::ro(accounts.quote_mint),
        AccountMeta::ro(accounts.base_token_program),
        AccountMeta::ro(accounts.quote_token_program),
        AccountMeta::ro(accounts.associated_token_program),
        AccountMeta::rw(accounts.fee_recipient),
        AccountMeta::rw(accounts.associated_quote_fee_recipient),
        AccountMeta::rw(accounts.buyback_fee_recipient),
        AccountMeta::rw(accounts.associated_quote_buyback_fee_recipient),
        AccountMeta::rw(accounts.bonding_curve),
        AccountMeta::rw(accounts.associated_base_bonding_curve),
        AccountMeta::rw(accounts.associated_quote_bonding_curve),
        AccountMeta::signer(accounts.user, true),
        AccountMeta::rw(accounts.associated_base_user),
        AccountMeta::rw(accounts.associated_quote_user),
        AccountMeta::rw(accounts.creator_vault),
        AccountMeta::rw(accounts.associated_creator_vault),
        AccountMeta::ro(accounts.sharing_config),
        AccountMeta::ro(accounts.global_volume_accumulator),
        AccountMeta::rw(accounts.user_volume_accumulator),
        AccountMeta::rw(accounts.associated_user_volume_accumulator),
        AccountMeta::ro(accounts.fee_config),
        AccountMeta::ro(accounts.fee_program),
        AccountMeta::ro(accounts.system_program),
        AccountMeta::ro(accounts.event_authority),
        AccountMeta::ro(accounts.program),
    ];
    debug_assert_eq!(metas.len(), BUY_V2_ACCOUNTS);
    Instruction {
        program_id: accounts.program,
        accounts: metas,
        data: encode_args(anchor_instruction_discriminator("buy_v2"), amount, max_sol_cost),
    }
}

/// `sell_v2(amount, min_sol_output)`.
pub fn build_sell_v2(accounts: &SellAccounts, amount: u64, min_sol_output: u64) -> Instruction {
    let metas = vec![
        AccountMeta::ro(accounts.global),
        AccountMeta::ro(accounts.base_mint),
        AccountMeta::ro(accounts.quote_mint),
        AccountMeta::ro(accounts.base_token_program),
        AccountMeta::ro(accounts.quote_token_program),
        AccountMeta::ro(accounts.associated_token_program),
        AccountMeta::rw(accounts.fee_recipient),
        AccountMeta::rw(accounts.associated_quote_fee_recipient),
        AccountMeta::rw(accounts.buyback_fee_recipient),
        AccountMeta::rw(accounts.associated_quote_buyback_fee_recipient),
        AccountMeta::rw(accounts.bonding_curve),
        AccountMeta::rw(accounts.associated_base_bonding_curve),
        AccountMeta::rw(accounts.associated_quote_bonding_curve),
        // Writable and signer on BOTH instructions. The prose table said
        // signer-only here; the IDL says otherwise and the IDL is what the
        // program was compiled against.
        AccountMeta::signer(accounts.user, true),
        AccountMeta::rw(accounts.associated_base_user),
        AccountMeta::rw(accounts.associated_quote_user),
        AccountMeta::rw(accounts.creator_vault),
        AccountMeta::rw(accounts.associated_creator_vault),
        AccountMeta::ro(accounts.sharing_config),
        AccountMeta::rw(accounts.user_volume_accumulator),
        AccountMeta::rw(accounts.associated_user_volume_accumulator),
        AccountMeta::ro(accounts.fee_config),
        AccountMeta::ro(accounts.fee_program),
        AccountMeta::ro(accounts.system_program),
        AccountMeta::ro(accounts.event_authority),
        AccountMeta::ro(accounts.program),
    ];
    debug_assert_eq!(metas.len(), SELL_V2_ACCOUNTS);
    Instruction {
        program_id: accounts.program,
        accounts: metas,
        data: encode_args(anchor_instruction_discriminator("sell_v2"), amount, min_sol_output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pk(seed: u8) -> Pubkey {
        [seed; PUBKEY_LEN]
    }

    fn buy_accounts() -> BuyAccounts {
        BuyAccounts {
            global: pk(1), base_mint: pk(2), quote_mint: pk(3),
            base_token_program: pk(4), quote_token_program: pk(5),
            associated_token_program: pk(6), fee_recipient: pk(7),
            associated_quote_fee_recipient: pk(8), buyback_fee_recipient: pk(9),
            associated_quote_buyback_fee_recipient: pk(10), bonding_curve: pk(11),
            associated_base_bonding_curve: pk(12), associated_quote_bonding_curve: pk(13),
            user: pk(14), associated_base_user: pk(15), associated_quote_user: pk(16),
            creator_vault: pk(17), associated_creator_vault: pk(18), sharing_config: pk(19),
            global_volume_accumulator: pk(20), user_volume_accumulator: pk(21),
            associated_user_volume_accumulator: pk(22), fee_config: pk(23),
            fee_program: pk(24), system_program: pk(25), event_authority: pk(26),
            program: pk(27),
        }
    }

    fn sell_accounts() -> SellAccounts {
        SellAccounts {
            global: pk(1), base_mint: pk(2), quote_mint: pk(3),
            base_token_program: pk(4), quote_token_program: pk(5),
            associated_token_program: pk(6), fee_recipient: pk(7),
            associated_quote_fee_recipient: pk(8), buyback_fee_recipient: pk(9),
            associated_quote_buyback_fee_recipient: pk(10), bonding_curve: pk(11),
            associated_base_bonding_curve: pk(12), associated_quote_bonding_curve: pk(13),
            user: pk(14), associated_base_user: pk(15), associated_quote_user: pk(16),
            creator_vault: pk(17), associated_creator_vault: pk(18), sharing_config: pk(19),
            user_volume_accumulator: pk(20), associated_user_volume_accumulator: pk(21),
            fee_config: pk(22), fee_program: pk(23), system_program: pk(24),
            event_authority: pk(25), program: pk(26),
        }
    }

    #[test]
    fn buy_has_the_documented_account_count_and_order() {
        let ix = build_buy_v2(&buy_accounts(), 1_000, 2_000);
        assert_eq!(ix.accounts.len(), BUY_V2_ACCOUNTS);
        let order: Vec<u8> = ix.accounts.iter().map(|meta| meta.pubkey[0]).collect();
        assert_eq!(order, (1..=27).collect::<Vec<u8>>());
    }

    #[test]
    fn sell_has_the_documented_account_count_and_order() {
        let ix = build_sell_v2(&sell_accounts(), 1_000, 2_000);
        assert_eq!(ix.accounts.len(), SELL_V2_ACCOUNTS);
        let order: Vec<u8> = ix.accounts.iter().map(|meta| meta.pubkey[0]).collect();
        assert_eq!(order, (1..=26).collect::<Vec<u8>>());
    }

    #[test]
    fn user_is_the_only_signer_and_is_writable_on_both() {
        // The prose table said signer-only on a sell. The IDL says writable on
        // both, and the IDL is what the program was compiled against.
        let buy = build_buy_v2(&buy_accounts(), 1, 1);
        let sell = build_sell_v2(&sell_accounts(), 1, 1);
        for (ix, signers) in [
            (&buy, &crate::generated_flags::BUY_V2_SIGNERS[..]),
            (&sell, &crate::generated_flags::SELL_V2_SIGNERS[..]),
        ] {
            let found: Vec<usize> = ix
                .accounts
                .iter()
                .enumerate()
                .filter(|(_, meta)| meta.is_signer)
                .map(|(index, _)| index + 1)
                .collect();
            assert_eq!(found, signers.to_vec());
            assert!(ix.accounts[signers[0] - 1].is_writable);
        }
    }

    #[test]
    fn only_the_documented_accounts_are_writable_on_a_buy() {
        let ix = build_buy_v2(&buy_accounts(), 1, 1);
        let writable: Vec<usize> = ix
            .accounts
            .iter()
            .enumerate()
            .filter(|(_, meta)| meta.is_writable)
            .map(|(index, _)| index + 1)
            .collect();
        // Generated from idl/pump.json, so a hand edit here cannot drift from
        // what the program actually expects.
        assert_eq!(writable, crate::generated_flags::BUY_V2_WRITABLE.to_vec());
        assert_eq!(ix.accounts.len(), crate::generated_flags::BUY_V2_ACCOUNT_COUNT);
    }

    #[test]
    fn only_the_documented_accounts_are_writable_on_a_sell() {
        let ix = build_sell_v2(&sell_accounts(), 1, 1);
        let writable: Vec<usize> = ix
            .accounts
            .iter()
            .enumerate()
            .filter(|(_, meta)| meta.is_writable)
            .map(|(index, _)| index + 1)
            .collect();
        assert_eq!(writable, crate::generated_flags::SELL_V2_WRITABLE.to_vec());
        assert_eq!(ix.accounts.len(), crate::generated_flags::SELL_V2_ACCOUNT_COUNT);
    }

    #[test]
    fn arguments_are_little_endian_after_an_eight_byte_discriminator() {
        let ix = build_buy_v2(&buy_accounts(), 0x0102030405060708, 0x1112131415161718);
        assert_eq!(ix.data.len(), 24);
        assert_eq!(&ix.data[..8], &anchor_instruction_discriminator("buy_v2"));
        assert_eq!(&ix.data[8..16], &0x0102030405060708u64.to_le_bytes());
        assert_eq!(&ix.data[16..24], &0x1112131415161718u64.to_le_bytes());
    }

    #[test]
    fn buy_and_sell_discriminators_differ() {
        assert_ne!(
            anchor_instruction_discriminator("buy_v2"),
            anchor_instruction_discriminator("sell_v2")
        );
    }

    #[test]
    fn the_account_discriminator_matches_the_deployed_bonding_curve() {
        assert_eq!(
            anchor_account_discriminator("BondingCurve"),
            crate::curve::BONDING_CURVE_DISCRIMINATOR
        );
    }
}
