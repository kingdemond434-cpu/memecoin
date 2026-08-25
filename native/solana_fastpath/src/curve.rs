//! Bonding-curve decode, pricing and capacity, with no allocation on the hot path.
//!
//! This is the stage that runs on every curve update for every live token, so
//! it is the stage where per-event cost actually compounds. Decoding is a
//! borrow over the account bytes rather than a parse into owned values, and
//! the arithmetic is integer throughout.
//!
//! Every product is taken in `u128`. The reserves are `u64` and a Pump curve
//! routinely holds ~10^12 base units against ~10^10 lamports, so
//! `virtual_token * amount` overflows `u64` on ordinary trades -- not on
//! adversarial ones. In release builds that overflow wraps silently and the
//! quote comes back plausible and wrong, which is the worst failure shape
//! available to a pricing function.

/// `sha256("account:BondingCurve")[..8]`, verified against the deployed program.
pub const BONDING_CURVE_DISCRIMINATOR: [u8; 8] = [23, 183, 248, 55, 96, 216, 172, 96];

/// Pump's published pre-2026-09-01 flat trade fee.
pub const LEGACY_FEE_BPS: u64 = 100;

pub const LAMPORTS_PER_SOL: u64 = 1_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BondingCurve {
    pub virtual_token_reserves: u64,
    pub virtual_sol_reserves: u64,
    pub real_token_reserves: u64,
    pub real_sol_reserves: u64,
    pub token_total_supply: u64,
    pub complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteError {
    NonPositiveInput,
    CurveCompleteOrEmpty,
    ConsumedByFee,
    RoundsToZero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quote {
    pub input_amount: u64,
    pub output_amount: u64,
    pub fee_amount: u64,
    /// Impact in basis points against the pre-trade spot price.
    pub price_impact_bps: u64,
}

fn read_u64(data: &[u8], offset: usize) -> Option<u64> {
    data.get(offset..offset + 8)
        .map(|slice| u64::from_le_bytes(slice.try_into().expect("slice is 8 bytes")))
}

impl BondingCurve {
    /// Decode from raw account data, borrowing rather than copying.
    ///
    /// Returns `None` on a discriminator mismatch rather than decoding
    /// anyway: an account of a different type whose bytes happen to parse
    /// yields a curve that prices confidently and wrongly.
    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 49 || data[..8] != BONDING_CURVE_DISCRIMINATOR {
            return None;
        }
        Some(Self {
            virtual_token_reserves: read_u64(data, 8)?,
            virtual_sol_reserves: read_u64(data, 16)?,
            real_token_reserves: read_u64(data, 24)?,
            real_sol_reserves: read_u64(data, 32)?,
            token_total_supply: read_u64(data, 40)?,
            complete: *data.get(48)? != 0,
        })
    }

    pub fn tradeable(&self) -> bool {
        !self.complete && self.virtual_token_reserves > 0 && self.virtual_sol_reserves > 0
    }

    /// Spot price in lamports per token, scaled by 1e9 to stay integral.
    pub fn spot_price_scaled(&self) -> Option<u128> {
        if self.virtual_token_reserves == 0 {
            return None;
        }
        Some(
            (self.virtual_sol_reserves as u128 * LAMPORTS_PER_SOL as u128)
                / self.virtual_token_reserves as u128,
        )
    }

    /// Tokens received for spending `lamports`, net of the quote-leg fee.
    pub fn quote_buy(&self, lamports: u64, fee_bps: u64) -> Result<Quote, QuoteError> {
        if lamports == 0 {
            return Err(QuoteError::NonPositiveInput);
        }
        if !self.tradeable() {
            return Err(QuoteError::CurveCompleteOrEmpty);
        }
        let fee = (lamports as u128 * fee_bps as u128 / 10_000) as u64;
        let net = lamports.checked_sub(fee).ok_or(QuoteError::ConsumedByFee)?;
        if net == 0 {
            return Err(QuoteError::ConsumedByFee);
        }

        // k = virtual_sol * virtual_token held constant across the trade.
        let numerator = net as u128 * self.virtual_token_reserves as u128;
        let denominator = self.virtual_sol_reserves as u128 + net as u128;
        let mut tokens_out = (numerator / denominator) as u64;
        if tokens_out == 0 {
            return Err(QuoteError::RoundsToZero);
        }
        // The curve cannot deliver more inventory than it holds.
        if self.real_token_reserves > 0 && tokens_out > self.real_token_reserves {
            tokens_out = self.real_token_reserves;
        }

        Ok(Quote {
            input_amount: lamports,
            output_amount: tokens_out,
            fee_amount: fee,
            price_impact_bps: self.buy_impact_bps(net, tokens_out),
        })
    }

    fn buy_impact_bps(&self, net_lamports: u64, tokens_out: u64) -> u64 {
        let Some(spot) = self.spot_price_scaled() else {
            return 0;
        };
        if spot == 0 || tokens_out == 0 {
            return 0;
        }
        let average = (net_lamports as u128 * LAMPORTS_PER_SOL as u128) / tokens_out as u128;
        if average <= spot {
            return 0;
        }
        (((average - spot) * 10_000) / spot) as u64
    }

    /// Lamports received for selling `tokens`, net of the quote-leg fee.
    pub fn quote_sell(&self, tokens: u64, fee_bps: u64) -> Result<Quote, QuoteError> {
        if tokens == 0 {
            return Err(QuoteError::NonPositiveInput);
        }
        if !self.tradeable() {
            return Err(QuoteError::CurveCompleteOrEmpty);
        }
        let numerator = tokens as u128 * self.virtual_sol_reserves as u128;
        let denominator = self.virtual_token_reserves as u128 + tokens as u128;
        let mut gross = (numerator / denominator) as u64;
        if gross == 0 {
            return Err(QuoteError::RoundsToZero);
        }
        // Only real SOL can actually be paid out, whatever the virtual curve says.
        if self.real_sol_reserves > 0 && gross > self.real_sol_reserves {
            gross = self.real_sol_reserves;
        }
        let fee = (gross as u128 * fee_bps as u128 / 10_000) as u64;
        let net = gross.checked_sub(fee).ok_or(QuoteError::ConsumedByFee)?;
        if net == 0 {
            return Err(QuoteError::ConsumedByFee);
        }

        Ok(Quote {
            input_amount: tokens,
            output_amount: net,
            fee_amount: fee,
            price_impact_bps: self.sell_impact_bps(net, tokens),
        })
    }

    fn sell_impact_bps(&self, net_lamports: u64, tokens: u64) -> u64 {
        let Some(spot) = self.spot_price_scaled() else {
            return 0;
        };
        if spot == 0 || tokens == 0 {
            return 0;
        }
        let average = (net_lamports as u128 * LAMPORTS_PER_SOL as u128) / tokens as u128;
        if average >= spot {
            return 0;
        }
        (((spot - average) * 10_000) / spot) as u64
    }

    /// Largest sale, in tokens, whose impact stays within `max_impact_bps`.
    ///
    /// The search starts from zero and never requires its low endpoint to
    /// satisfy the bound. Impact is monotone in size only above the
    /// integer-rounding region: on a real curve a 64-unit sell reports far
    /// higher impact than a 1,000,000-unit sell, purely from quantisation.
    /// Demanding that the low endpoint qualify would let one artefact at the
    /// bottom of the range report the whole curve as unexecutable.
    pub fn sell_capacity(&self, max_impact_bps: u64, fee_bps: u64) -> u64 {
        if !self.tradeable() || max_impact_bps == 0 {
            return 0;
        }
        let ceiling = self.real_token_reserves.max(self.virtual_token_reserves);
        if ceiling == 0 {
            return 0;
        }
        let (mut low, mut high) = (0u64, ceiling);
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            match self.quote_sell(mid, fee_bps) {
                Ok(quote) if quote.price_impact_bps <= max_impact_bps => low = mid,
                _ => high = mid - 1,
            }
        }
        low
    }

    /// Executable size at each impact bound, for the exit side.
    pub fn exit_frontier(&self, bounds: &[u64], fee_bps: u64) -> Vec<(u64, u64)> {
        bounds
            .iter()
            .map(|&bound| (bound, self.sell_capacity(bound, fee_bps)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve() -> BondingCurve {
        BondingCurve {
            virtual_token_reserves: 1_073_000_000_000_000,
            virtual_sol_reserves: 30_000_000_000,
            real_token_reserves: 793_100_000_000_000,
            real_sol_reserves: 5_000_000_000,
            token_total_supply: 1_000_000_000_000_000,
            complete: false,
        }
    }

    fn encode(curve: &BondingCurve) -> Vec<u8> {
        let mut data = BONDING_CURVE_DISCRIMINATOR.to_vec();
        for value in [
            curve.virtual_token_reserves,
            curve.virtual_sol_reserves,
            curve.real_token_reserves,
            curve.real_sol_reserves,
            curve.token_total_supply,
        ] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        data.push(curve.complete as u8);
        data
    }

    #[test]
    fn decodes_a_real_layout() {
        let original = curve();
        assert_eq!(BondingCurve::decode(&encode(&original)), Some(original));
    }

    #[test]
    fn rejects_a_foreign_discriminator() {
        let mut data = encode(&curve());
        data[0] ^= 0xff;
        // An account of another type whose bytes happen to parse would price
        // confidently and wrongly.
        assert_eq!(BondingCurve::decode(&data), None);
    }

    #[test]
    fn rejects_truncated_data() {
        let data = encode(&curve());
        assert_eq!(BondingCurve::decode(&data[..40]), None);
    }

    #[test]
    fn a_complete_curve_is_not_tradeable() {
        let mut done = curve();
        done.complete = true;
        assert!(!done.tradeable());
        assert_eq!(done.quote_buy(1_000_000, LEGACY_FEE_BPS), Err(QuoteError::CurveCompleteOrEmpty));
    }

    #[test]
    fn large_trades_do_not_overflow() {
        // virtual_token * amount here exceeds u64 by orders of magnitude; in
        // release builds a u64 product would wrap and return a plausible,
        // wrong quote.
        let quote = curve().quote_buy(50_000_000_000, LEGACY_FEE_BPS).unwrap();
        assert!(quote.output_amount > 0);
        assert!(quote.output_amount < curve().real_token_reserves);
    }

    #[test]
    fn impact_rises_with_size() {
        let small = curve().quote_buy(100_000_000, LEGACY_FEE_BPS).unwrap();
        let large = curve().quote_buy(10_000_000_000, LEGACY_FEE_BPS).unwrap();
        assert!(large.price_impact_bps > small.price_impact_bps);
    }

    #[test]
    fn the_fee_is_taken_from_the_quote_leg() {
        let quote = curve().quote_buy(1_000_000_000, LEGACY_FEE_BPS).unwrap();
        assert_eq!(quote.fee_amount, 10_000_000);
    }

    #[test]
    fn a_sell_cannot_exceed_real_sol_reserves() {
        let quote = curve().quote_sell(900_000_000_000_000, LEGACY_FEE_BPS).unwrap();
        assert!(quote.output_amount + quote.fee_amount <= curve().real_sol_reserves);
    }

    #[test]
    fn capacity_is_monotone_in_the_bound() {
        let c = curve();
        let sizes: Vec<u64> = [100u64, 300, 500, 1_000]
            .iter()
            .map(|&bps| c.sell_capacity(bps, LEGACY_FEE_BPS))
            .collect();
        let mut sorted = sizes.clone();
        sorted.sort_unstable();
        assert_eq!(sizes, sorted);
        assert!(sizes[0] > 0);
    }

    #[test]
    fn capacity_respects_its_own_bound() {
        let c = curve();
        let size = c.sell_capacity(500, LEGACY_FEE_BPS);
        assert!(c.quote_sell(size, LEGACY_FEE_BPS).unwrap().price_impact_bps <= 500);
        // And one unit past it does not.
        assert!(c.quote_sell(size + 1, LEGACY_FEE_BPS).unwrap().price_impact_bps > 500);
    }

    #[test]
    fn a_rounding_artefact_at_the_bottom_does_not_zero_the_frontier() {
        let c = curve();
        // A tiny sell reports outsized impact purely from quantisation; the
        // frontier must not conclude from it that nothing is executable.
        assert!(c.quote_sell(1, LEGACY_FEE_BPS).is_err()
            || c.quote_sell(1, LEGACY_FEE_BPS).unwrap().price_impact_bps > 100);
        assert!(c.sell_capacity(100, LEGACY_FEE_BPS) > 1_000_000);
    }

    #[test]
    fn zero_input_is_rejected_on_both_sides() {
        assert_eq!(curve().quote_buy(0, LEGACY_FEE_BPS), Err(QuoteError::NonPositiveInput));
        assert_eq!(curve().quote_sell(0, LEGACY_FEE_BPS), Err(QuoteError::NonPositiveInput));
    }

    #[test]
    fn a_fee_that_consumes_the_input_is_rejected() {
        assert_eq!(curve().quote_buy(10, 10_000), Err(QuoteError::ConsumedByFee));
    }
}
