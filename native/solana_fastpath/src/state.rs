//! Per-token hot state, updated from decoded events.
//!
//! The decision path needs a handful of facts about a token and needs them
//! without allocating, without a lock and without asking anything over a
//! network: the current reserves, who bought and in what order, whether the
//! creator has sold, how old the launch is. Everything else -- the lifetime
//! wallet histories, the source genealogy, the trained artifacts -- is a cold
//! moat consulted off the hot path and summarised into the few numbers here.
//!
//! Two design choices carry most of the weight.
//!
//! The buyer sequence is a fixed array rather than a growing vector. The
//! First-25 cohort is the only part of the buyer list a T0 decision reads, so
//! the state holds exactly that many and counts the rest. A per-token vector
//! that grows with a viral mint is an allocation on the hot path and a memory
//! leak on a quiet day, and neither buys anything the twenty-sixth buyer's
//! identity would have told us.
//!
//! And every field that could be unobserved is explicitly optional. A launch
//! whose creator we have not identified is not a launch with creator zero:
//! the difference decides whether a transaction can be built at all, and
//! collapsing it into a default is how a desk ends up deriving a vault for
//! nobody.

pub const FIRST_COHORT: usize = 25;

/// One buyer, in arrival order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Buyer {
    /// Wallet skill where the cold moat has scored it. `None` means never
    /// seen, which is emphatically not the same as scored zero: a wave of
    /// unknown wallets is not a wave of bad ones.
    pub skill: Option<f64>,
    pub notional_sol: f64,
    pub at_seconds: f64,
    pub is_creator_linked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reserves {
    pub virtual_sol: u64,
    pub virtual_token: u64,
    /// Zero until a full account update supplies them. That is what marks a
    /// frontier derived from this state as an upper bound rather than a
    /// measurement.
    pub real_sol: u64,
    pub real_token: u64,
    pub measured: bool,
}

impl Reserves {
    pub fn tradeable(&self) -> bool {
        self.virtual_sol > 0 && self.virtual_token > 0
    }

    /// Quote-side depth in lamports: what a seller can actually be paid out
    /// of. Real where measured, virtual as an upper bound otherwise.
    pub fn depth_lamports(&self) -> u64 {
        if self.measured && self.real_sol > 0 {
            self.real_sol
        } else {
            self.virtual_sol
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenState {
    pub created_at: f64,
    pub last_event_at: f64,
    pub reserves: Reserves,
    pub creator: Option<[u8; 32]>,
    buyers: [Option<Buyer>; FIRST_COHORT],
    buyer_count: u32,
    pub sell_count: u32,
    pub creator_sold: bool,
    pub buy_notional_sol: f64,
    pub sell_notional_sol: f64,
    /// Seconds between the first public mention we saw and the launch. None
    /// when no source named it, which is different from "named at zero lag".
    pub first_source_lag: Option<f64>,
}

impl TokenState {
    pub fn new(created_at: f64) -> Self {
        Self {
            created_at,
            last_event_at: created_at,
            reserves: Reserves {
                virtual_sol: 0,
                virtual_token: 0,
                real_sol: 0,
                real_token: 0,
                measured: false,
            },
            creator: None,
            buyers: [None; FIRST_COHORT],
            buyer_count: 0,
            sell_count: 0,
            creator_sold: false,
            buy_notional_sol: 0.0,
            sell_notional_sol: 0.0,
            first_source_lag: None,
        }
    }

    pub fn age_seconds(&self, now: f64) -> f64 {
        (now - self.created_at).max(0.0)
    }

    pub fn observed_buyers(&self) -> u32 {
        self.buyer_count
    }

    pub fn cohort(&self) -> &[Option<Buyer>; FIRST_COHORT] {
        &self.buyers
    }

    /// Record a buy. Only the first `FIRST_COHORT` are retained by identity;
    /// the rest advance the counters, because the twenty-sixth buyer changes
    /// the flow numbers and not the cohort's composition.
    pub fn record_buy(&mut self, buyer: Buyer) {
        if (self.buyer_count as usize) < FIRST_COHORT {
            self.buyers[self.buyer_count as usize] = Some(buyer);
        }
        self.buyer_count = self.buyer_count.saturating_add(1);
        self.buy_notional_sol += buyer.notional_sol.max(0.0);
        self.last_event_at = self.last_event_at.max(buyer.at_seconds);
    }

    pub fn record_sell(&mut self, notional_sol: f64, at_seconds: f64, creator_linked: bool) {
        self.sell_count = self.sell_count.saturating_add(1);
        self.sell_notional_sol += notional_sol.max(0.0);
        self.creator_sold |= creator_linked;
        self.last_event_at = self.last_event_at.max(at_seconds);
    }

    pub fn update_reserves(&mut self, reserves: Reserves, at_seconds: f64) {
        // A measured update never loses to a reconstructed one. Merging them
        // would produce a record that is neither, and nothing downstream
        // could tell which fields to trust.
        if self.reserves.measured && !reserves.measured {
            self.reserves.virtual_sol = reserves.virtual_sol;
            self.reserves.virtual_token = reserves.virtual_token;
        } else {
            self.reserves = reserves;
        }
        self.last_event_at = self.last_event_at.max(at_seconds);
    }

    /// Mean skill across scored buyers in the cohort, and how many were
    /// scored. Returning the count alongside is what stops a mean over two
    /// wallets being read with the confidence of a mean over twenty-five.
    pub fn cohort_skill(&self) -> (Option<f64>, u32) {
        let mut total = 0.0;
        let mut scored = 0u32;
        for slot in self.buyers.iter().flatten() {
            if let Some(skill) = slot.skill {
                total += skill;
                scored += 1;
            }
        }
        if scored == 0 {
            (None, 0)
        } else {
            (Some(total / scored as f64), scored)
        }
    }

    /// Share of the cohort that is linked to the creator's own funding tree.
    /// None when the cohort is empty rather than zero, because an empty
    /// cohort is not an independent one.
    pub fn creator_linked_share(&self) -> Option<f64> {
        let seen = self.buyers.iter().flatten().count();
        if seen == 0 {
            return None;
        }
        let linked = self
            .buyers
            .iter()
            .flatten()
            .filter(|buyer| buyer.is_creator_linked)
            .count();
        Some(linked as f64 / seen as f64)
    }

    /// Buys per second across the cohort window. None before two buyers,
    /// since one arrival has no rate.
    pub fn buy_velocity(&self) -> Option<f64> {
        let mut first = f64::MAX;
        let mut last = f64::MIN;
        let mut seen = 0u32;
        for buyer in self.buyers.iter().flatten() {
            first = first.min(buyer.at_seconds);
            last = last.max(buyer.at_seconds);
            seen += 1;
        }
        if seen < 2 {
            return None;
        }
        let span = last - first;
        if span <= 0.0 {
            // Every buy in one slot. A real and important shape -- it is what
            // a bundle looks like -- but a rate over zero time is not a
            // number, so it is reported as unmeasurable rather than infinite.
            return None;
        }
        Some((seen - 1) as f64 / span)
    }

    /// Net SOL into the token. Negative when sellers dominate.
    pub fn net_flow_sol(&self) -> f64 {
        self.buy_notional_sol - self.sell_notional_sol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buyer(skill: Option<f64>, at: f64) -> Buyer {
        Buyer {
            skill,
            notional_sol: 1.0,
            at_seconds: at,
            is_creator_linked: false,
        }
    }

    #[test]
    fn the_cohort_is_bounded_and_the_rest_still_counts() {
        let mut state = TokenState::new(0.0);
        for index in 0..60 {
            state.record_buy(buyer(Some(0.5), index as f64 * 0.01));
        }
        assert_eq!(state.observed_buyers(), 60);
        assert_eq!(state.cohort().iter().flatten().count(), FIRST_COHORT);
        assert!((state.buy_notional_sol - 60.0).abs() < 1e-9);
    }

    #[test]
    fn an_unscored_wallet_is_not_a_zero_scored_one() {
        let mut state = TokenState::new(0.0);
        state.record_buy(buyer(None, 0.0));
        state.record_buy(buyer(None, 0.1));
        assert_eq!(state.cohort_skill(), (None, 0));
        state.record_buy(buyer(Some(0.8), 0.2));
        let (mean, scored) = state.cohort_skill();
        assert_eq!(scored, 1);
        assert!((mean.unwrap() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn an_empty_cohort_has_no_linked_share() {
        let state = TokenState::new(0.0);
        assert!(state.creator_linked_share().is_none());
    }

    #[test]
    fn one_buyer_has_no_velocity_and_one_slot_is_unmeasurable() {
        let mut state = TokenState::new(0.0);
        state.record_buy(buyer(Some(0.5), 1.0));
        assert!(state.buy_velocity().is_none());
        state.record_buy(buyer(Some(0.5), 1.0));
        // Same instant: a real shape, and still not a rate.
        assert!(state.buy_velocity().is_none());
        state.record_buy(buyer(Some(0.5), 2.0));
        assert!(state.buy_velocity().unwrap() > 0.0);
    }

    #[test]
    fn a_reconstruction_never_overwrites_a_measurement() {
        let mut state = TokenState::new(0.0);
        state.update_reserves(
            Reserves {
                virtual_sol: 30,
                virtual_token: 1_000,
                real_sol: 12,
                real_token: 800,
                measured: true,
            },
            1.0,
        );
        state.update_reserves(
            Reserves {
                virtual_sol: 31,
                virtual_token: 990,
                real_sol: 0,
                real_token: 0,
                measured: false,
            },
            2.0,
        );
        assert!(state.reserves.measured);
        assert_eq!(state.reserves.real_sol, 12);
        // The virtual side still tracks the newer event.
        assert_eq!(state.reserves.virtual_sol, 31);
    }

    #[test]
    fn depth_prefers_the_measured_reserve() {
        let mut reserves = Reserves {
            virtual_sol: 30,
            virtual_token: 1_000,
            real_sol: 0,
            real_token: 0,
            measured: false,
        };
        assert_eq!(reserves.depth_lamports(), 30);
        reserves.real_sol = 12;
        reserves.measured = true;
        assert_eq!(reserves.depth_lamports(), 12);
    }

    #[test]
    fn net_flow_goes_negative_when_sellers_dominate() {
        let mut state = TokenState::new(0.0);
        state.record_buy(buyer(Some(0.5), 0.0));
        state.record_sell(5.0, 1.0, true);
        assert!(state.net_flow_sol() < 0.0);
        assert!(state.creator_sold);
    }
}
