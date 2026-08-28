//! PumpSwap pool decode, pricing and capacity.
//!
//! A token that graduates off the bonding curve must not become a new token.
//! Rediscovering it after migration throws away the actor graph, the monster
//! state, the hazard state and the position history built over its whole life
//! -- exactly at the moment that history is most informative. So the pool is
//! priced by the same shape of arithmetic as the curve, and the state machine
//! above carries across the boundary rather than restarting behind it.
//!
//! The ordered buy/sell account lists and flags are generated from the
//! vendored official Pump AMM IDL. They are not transcribed from prose and are
//! guarded by the same regeneration test as the Pump curve instructions.

use crate::generated_flags::{
    PUMPSWAP_BUY_ACCOUNT_COUNT, PUMPSWAP_BUY_SIGNERS, PUMPSWAP_BUY_WRITABLE,
    PUMPSWAP_SELL_ACCOUNT_COUNT, PUMPSWAP_SELL_SIGNERS, PUMPSWAP_SELL_WRITABLE,
};
use crate::instruction::{AccountMeta, Instruction, Pubkey};

/// `sha256("account:Pool")[..8]`.
pub const POOL_DISCRIMINATOR: [u8; 8] = [241, 154, 109, 4, 17, 177, 109, 188];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pool {
    pub pool_bump: u8,
    pub index: u16,
    pub creator: [u8; 32],
    pub base_mint: [u8; 32],
    pub quote_mint: [u8; 32],
    pub lp_mint: [u8; 32],
    pub pool_base_token_account: [u8; 32],
    pub pool_quote_token_account: [u8; 32],
    pub lp_supply: u64,
    pub coin_creator: [u8; 32],
    pub is_mayhem_mode: bool,
    pub is_cashback_coin: bool,
    /// Signed in the published layout. Treated as signed rather than coerced:
    /// a negative value is a state this code does not understand, and reading
    /// it as a huge positive reserve would price a trade against liquidity
    /// that is not there.
    pub virtual_quote_reserves: i128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolQuoteError {
    NonPositiveInput,
    EmptyReserves,
    NegativeVirtualReserves,
    ExceedsReserves,
    RoundsToZero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolQuote {
    pub input_amount: u64,
    pub output_amount: u64,
    pub fee_amount: u64,
    pub price_impact_bps: u64,
}

fn read_u64(data: &[u8], offset: usize) -> Option<u64> {
    data.get(offset..offset + 8)
        .map(|slice| u64::from_le_bytes(slice.try_into().expect("8 bytes")))
}

fn read_pubkey(data: &[u8], offset: usize) -> Option<[u8; 32]> {
    data.get(offset..offset + 32)
        .map(|slice| slice.try_into().expect("32 bytes"))
}

impl Pool {
    /// Byte length of the published layout after the 8-byte discriminator.
    pub const BODY_LEN: usize = 1 + 2 + 32 * 5 + 8 + 32 + 1 + 1 + 16;

    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 8 + Self::BODY_LEN || data[..8] != POOL_DISCRIMINATOR {
            return None;
        }
        let mut offset = 8;
        let pool_bump = *data.get(offset)?;
        offset += 1;
        let index = u16::from_le_bytes(data.get(offset..offset + 2)?.try_into().ok()?);
        offset += 2;
        let creator = read_pubkey(data, offset)?;
        offset += 32;
        let base_mint = read_pubkey(data, offset)?;
        offset += 32;
        let quote_mint = read_pubkey(data, offset)?;
        offset += 32;
        let lp_mint = read_pubkey(data, offset)?;
        offset += 32;
        let pool_base_token_account = read_pubkey(data, offset)?;
        offset += 32;
        let pool_quote_token_account = read_pubkey(data, offset)?;
        offset += 32;
        let lp_supply = read_u64(data, offset)?;
        offset += 8;
        let coin_creator = read_pubkey(data, offset)?;
        offset += 32;
        let is_mayhem_mode = *data.get(offset)? != 0;
        offset += 1;
        let is_cashback_coin = *data.get(offset)? != 0;
        offset += 1;
        let virtual_quote_reserves =
            i128::from_le_bytes(data.get(offset..offset + 16)?.try_into().ok()?);

        Some(Self {
            pool_bump,
            index,
            creator,
            base_mint,
            quote_mint,
            lp_mint,
            pool_base_token_account,
            pool_quote_token_account,
            lp_supply,
            coin_creator,
            is_mayhem_mode,
            is_cashback_coin,
            virtual_quote_reserves,
        })
    }
}

/// Reserves as observed from the pool's own token accounts.
///
/// Held separately from `Pool` because the vault balances are not in the pool
/// account: pricing needs both, and pretending otherwise would invite a quote
/// computed from whichever half happened to be available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolReserves {
    pub base: u64,
    pub quote: u64,
}

impl PoolReserves {
    fn spot_scaled(&self) -> Option<u128> {
        if self.base == 0 {
            return None;
        }
        Some((self.quote as u128 * 1_000_000_000u128) / self.base as u128)
    }

    /// Base tokens out for `quote_in`, constant product, fee on the quote leg.
    pub fn quote_buy(&self, quote_in: u64, fee_bps: u64) -> Result<PoolQuote, PoolQuoteError> {
        if quote_in == 0 {
            return Err(PoolQuoteError::NonPositiveInput);
        }
        if self.base == 0 || self.quote == 0 {
            return Err(PoolQuoteError::EmptyReserves);
        }
        let fee = (quote_in as u128 * fee_bps as u128 / 10_000) as u64;
        let net = quote_in.saturating_sub(fee);
        if net == 0 {
            return Err(PoolQuoteError::RoundsToZero);
        }
        // u128 throughout: base reserves routinely exceed 10^12 and the
        // product overflows u64 on ordinary size.
        let out = ((net as u128 * self.base as u128) / (self.quote as u128 + net as u128)) as u64;
        if out == 0 {
            return Err(PoolQuoteError::RoundsToZero);
        }
        if out >= self.base {
            return Err(PoolQuoteError::ExceedsReserves);
        }
        Ok(PoolQuote {
            input_amount: quote_in,
            output_amount: out,
            fee_amount: fee,
            price_impact_bps: self.impact_bps(net, out, true),
        })
    }

    /// Quote tokens out for `base_in`, net of fee.
    pub fn quote_sell(&self, base_in: u64, fee_bps: u64) -> Result<PoolQuote, PoolQuoteError> {
        if base_in == 0 {
            return Err(PoolQuoteError::NonPositiveInput);
        }
        if self.base == 0 || self.quote == 0 {
            return Err(PoolQuoteError::EmptyReserves);
        }
        let gross =
            ((base_in as u128 * self.quote as u128) / (self.base as u128 + base_in as u128)) as u64;
        if gross == 0 {
            return Err(PoolQuoteError::RoundsToZero);
        }
        if gross >= self.quote {
            return Err(PoolQuoteError::ExceedsReserves);
        }
        let fee = (gross as u128 * fee_bps as u128 / 10_000) as u64;
        let net = gross.saturating_sub(fee);
        if net == 0 {
            return Err(PoolQuoteError::RoundsToZero);
        }
        Ok(PoolQuote {
            input_amount: base_in,
            output_amount: net,
            fee_amount: fee,
            price_impact_bps: self.impact_bps(net, base_in, false),
        })
    }

    fn impact_bps(&self, quote_amount: u64, base_amount: u64, is_buy: bool) -> u64 {
        let Some(spot) = self.spot_scaled() else {
            return 0;
        };
        if spot == 0 || base_amount == 0 {
            return 0;
        }
        let average = (quote_amount as u128 * 1_000_000_000u128) / base_amount as u128;
        let (high, low) = if is_buy {
            (average, spot)
        } else {
            (spot, average)
        };
        if high <= low {
            return 0;
        }
        (((high - low) * 10_000) / spot) as u64
    }

    /// Largest sale within an impact bound. Same search shape as the curve:
    /// starts at zero, never requires its low endpoint to qualify.
    pub fn sell_capacity(&self, max_impact_bps: u64, fee_bps: u64) -> u64 {
        if self.base == 0 || self.quote == 0 || max_impact_bps == 0 {
            return 0;
        }
        let (mut low, mut high) = (0u64, self.base.saturating_sub(1));
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            match self.quote_sell(mid, fee_bps) {
                Ok(quote) if quote.price_impact_bps <= max_impact_bps => low = mid,
                _ => high = mid - 1,
            }
        }
        low
    }
}

pub const ACCOUNT_LIST_STATUS: &str =
    "OK: PumpSwap account order and flags generated from idl/pump_amm.json";

/// `buy(base_out, max_quote_in)` instruction data.
///
pub fn buy_data(base_out: u64, max_quote_in: u64, track_volume: bool) -> Vec<u8> {
    let mut data = crate::instruction::anchor_instruction_discriminator("buy").to_vec();
    data.extend_from_slice(&base_out.to_le_bytes());
    data.extend_from_slice(&max_quote_in.to_le_bytes());
    // The IDL's OptionBool wrapper is one Borsh bool byte.
    data.push(u8::from(track_volume));
    data
}

/// `sell(base_in, min_quote_out)` instruction data.
pub fn sell_data(base_in: u64, min_quote_out: u64) -> Vec<u8> {
    let mut data = crate::instruction::anchor_instruction_discriminator("sell").to_vec();
    data.extend_from_slice(&base_in.to_le_bytes());
    data.extend_from_slice(&min_quote_out.to_le_bytes());
    data
}

fn build_metas(
    keys: &[Pubkey],
    expected: usize,
    writable: &[usize],
    signers: &[usize],
) -> Option<Vec<AccountMeta>> {
    if keys.len() != expected {
        return None;
    }
    Some(
        keys.iter()
            .enumerate()
            .map(|(offset, key)| {
                let position = offset + 1;
                AccountMeta {
                    pubkey: *key,
                    is_signer: signers.contains(&position),
                    is_writable: writable.contains(&position),
                }
            })
            .collect(),
    )
}

/// Complete PumpSwap buy instruction from the IDL-ordered account keys.
pub fn build_buy(
    keys: &[Pubkey],
    base_out: u64,
    max_quote_in: u64,
    track_volume: bool,
) -> Option<Instruction> {
    let accounts = build_metas(
        keys,
        PUMPSWAP_BUY_ACCOUNT_COUNT,
        &PUMPSWAP_BUY_WRITABLE,
        &PUMPSWAP_BUY_SIGNERS,
    )?;
    // `program` is account 17 in the published buy account order.
    Some(Instruction {
        program_id: keys[16],
        accounts,
        data: buy_data(base_out, max_quote_in, track_volume),
    })
}

/// Complete PumpSwap sell instruction from the IDL-ordered account keys.
pub fn build_sell(keys: &[Pubkey], base_in: u64, min_quote_out: u64) -> Option<Instruction> {
    let accounts = build_metas(
        keys,
        PUMPSWAP_SELL_ACCOUNT_COUNT,
        &PUMPSWAP_SELL_WRITABLE,
        &PUMPSWAP_SELL_SIGNERS,
    )?;
    // `program` is account 17 in the published sell account order too.
    Some(Instruction {
        program_id: keys[16],
        accounts,
        data: sell_data(base_in, min_quote_out),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> Pool {
        Pool {
            pool_bump: 254,
            index: 0,
            creator: [1; 32],
            base_mint: [2; 32],
            quote_mint: [3; 32],
            lp_mint: [4; 32],
            pool_base_token_account: [5; 32],
            pool_quote_token_account: [6; 32],
            lp_supply: 1_000_000,
            coin_creator: [7; 32],
            is_mayhem_mode: false,
            is_cashback_coin: true,
            virtual_quote_reserves: 42,
        }
    }

    fn encode(pool: &Pool) -> Vec<u8> {
        let mut data = POOL_DISCRIMINATOR.to_vec();
        data.push(pool.pool_bump);
        data.extend_from_slice(&pool.index.to_le_bytes());
        for key in [
            pool.creator,
            pool.base_mint,
            pool.quote_mint,
            pool.lp_mint,
            pool.pool_base_token_account,
            pool.pool_quote_token_account,
        ] {
            data.extend_from_slice(&key);
        }
        data.extend_from_slice(&pool.lp_supply.to_le_bytes());
        data.extend_from_slice(&pool.coin_creator);
        data.push(pool.is_mayhem_mode as u8);
        data.push(pool.is_cashback_coin as u8);
        data.extend_from_slice(&pool.virtual_quote_reserves.to_le_bytes());
        data
    }

    fn reserves() -> PoolReserves {
        PoolReserves {
            base: 200_000_000_000_000,
            quote: 85_000_000_000,
        }
    }

    #[test]
    fn decodes_the_published_layout() {
        assert_eq!(Pool::decode(&encode(&pool())), Some(pool()));
    }

    #[test]
    fn rejects_a_foreign_discriminator() {
        let mut data = encode(&pool());
        data[0] ^= 0xff;
        assert_eq!(Pool::decode(&data), None);
    }

    #[test]
    fn rejects_truncated_data() {
        let data = encode(&pool());
        assert_eq!(Pool::decode(&data[..data.len() - 1]), None);
    }

    #[test]
    fn the_signed_virtual_reserve_survives_a_negative_value() {
        let mut negative = pool();
        negative.virtual_quote_reserves = -1;
        // Coercing to unsigned would read this as an enormous reserve and
        // price a trade against liquidity that is not there.
        assert_eq!(
            Pool::decode(&encode(&negative))
                .unwrap()
                .virtual_quote_reserves,
            -1
        );
    }

    #[test]
    fn mayhem_and_cashback_flags_round_trip() {
        let mut flagged = pool();
        flagged.is_mayhem_mode = true;
        flagged.is_cashback_coin = false;
        let decoded = Pool::decode(&encode(&flagged)).unwrap();
        assert!(decoded.is_mayhem_mode);
        assert!(!decoded.is_cashback_coin);
    }

    #[test]
    fn large_trades_do_not_overflow() {
        let quote = reserves().quote_buy(10_000_000_000, 100).unwrap();
        assert!(quote.output_amount > 0);
        assert!(quote.output_amount < reserves().base);
    }

    #[test]
    fn impact_rises_with_size_on_both_sides() {
        let r = reserves();
        assert!(
            r.quote_buy(10_000_000_000, 100).unwrap().price_impact_bps
                > r.quote_buy(100_000_000, 100).unwrap().price_impact_bps
        );
        assert!(
            r.quote_sell(50_000_000_000_000, 100)
                .unwrap()
                .price_impact_bps
                > r.quote_sell(100_000_000, 100).unwrap().price_impact_bps
        );
    }

    #[test]
    fn an_empty_pool_prices_nothing() {
        let empty = PoolReserves { base: 0, quote: 0 };
        assert_eq!(
            empty.quote_buy(1_000, 100),
            Err(PoolQuoteError::EmptyReserves)
        );
        assert_eq!(
            empty.quote_sell(1_000, 100),
            Err(PoolQuoteError::EmptyReserves)
        );
        assert_eq!(empty.sell_capacity(500, 100), 0);
    }

    #[test]
    fn a_pool_cannot_be_drained_and_absurd_size_prices_itself_out() {
        let r = reserves();
        // Constant product approaches the reserve asymptotically and never
        // reaches it, so an enormous order still quotes -- correctly. What
        // rejects it is the impact, not a reserve guard, and the capacity
        // frontier is what consumes that.
        let quote = r.quote_buy(u64::MAX / 2, 100).unwrap();
        assert!(quote.output_amount < r.base);
        assert!(quote.price_impact_bps > 1_000_000);
        // And no sane bound admits a size anywhere near it. At a 10% impact
        // bound the frontier allows roughly a tenth of reserves, which is
        // vanishing next to an order that size.
        let allowed = r.sell_capacity(1_000, 100);
        assert!(allowed < r.base / 4);
        assert!((allowed as u128) < (u64::MAX / 2) as u128 / 1_000);
    }

    #[test]
    fn capacity_respects_its_bound() {
        let r = reserves();
        let size = r.sell_capacity(500, 100);
        assert!(size > 0);
        assert!(r.quote_sell(size, 100).unwrap().price_impact_bps <= 500);
        assert!(r.quote_sell(size + 1, 100).unwrap().price_impact_bps > 500);
    }

    #[test]
    fn capacity_is_monotone_in_the_bound() {
        let r = reserves();
        let sizes: Vec<u64> = [100u64, 300, 500, 1_000]
            .iter()
            .map(|&bps| r.sell_capacity(bps, 100))
            .collect();
        let mut sorted = sizes.clone();
        sorted.sort_unstable();
        assert_eq!(sizes, sorted);
    }

    #[test]
    fn instruction_data_is_published_arguments_only() {
        let buy = buy_data(0x0102030405060708, 0x1112131415161718, false);
        assert_eq!(buy.len(), 25);
        assert_eq!(&buy[8..16], &0x0102030405060708u64.to_le_bytes());
        assert_eq!(buy[24], 0);
        assert_eq!(buy_data(1, 1, true)[24], 1);
        assert_ne!(buy_data(1, 1, false)[..8], sell_data(1, 1)[..8]);
    }

    #[test]
    fn native_buy_uses_the_idl_generated_flags() {
        let keys: Vec<Pubkey> = (1..=PUMPSWAP_BUY_ACCOUNT_COUNT)
            .map(|index| [index as u8; 32])
            .collect();
        let instruction = build_buy(&keys, 10, 20, false).unwrap();
        assert_eq!(instruction.accounts.len(), PUMPSWAP_BUY_ACCOUNT_COUNT);
        assert_eq!(instruction.program_id, keys[16]);
        let writable: Vec<usize> = instruction
            .accounts
            .iter()
            .enumerate()
            .filter(|(_, meta)| meta.is_writable)
            .map(|(index, _)| index + 1)
            .collect();
        let signers: Vec<usize> = instruction
            .accounts
            .iter()
            .enumerate()
            .filter(|(_, meta)| meta.is_signer)
            .map(|(index, _)| index + 1)
            .collect();
        assert_eq!(writable, PUMPSWAP_BUY_WRITABLE);
        assert_eq!(signers, PUMPSWAP_BUY_SIGNERS);
        assert!(ACCOUNT_LIST_STATUS.starts_with("OK"));
    }

    #[test]
    fn native_sell_refuses_the_wrong_account_count() {
        let short = vec![[0u8; 32]; PUMPSWAP_SELL_ACCOUNT_COUNT - 1];
        assert!(build_sell(&short, 10, 20).is_none());
    }
}
