//! Event to decision, in one pass.
//!
//! The pieces existed separately: decode, price, score, check. Between them
//! sat Python, allocating an object per account and per bin on the path where
//! an event arrives and something has to be done about it. This joins them,
//! so the whole T0 sequence -- state, age band, executable bins, one Q per
//! action, hard safety -- runs without leaving Rust.
//!
//! What deliberately does NOT live here is inference. Calibrated probabilities
//! come from trained artifacts, and reimplementing gradient boosting in the
//! hot path would buy microseconds while introducing a second model that can
//! disagree with the one the promotion gate validated. The probabilities are
//! handed in; everything downstream of them is computed here.
//!
//! The decision carries its own age band and its own blocked reason. A
//! caller that receives a refusal should be able to say why without asking
//! anything else -- because the times that question gets asked are exactly
//! the times when nothing else is available to ask.

use crate::policy::{self, Action, Decision as PolicyDecision, Position, Survival};
use crate::safety::{self, Intent, Limits, Verdict};
use crate::state::TokenState;

#[derive(Debug, Clone)]
pub struct T0Decision {
    pub action: Action,
    pub q: f64,
    pub age_band: &'static str,
    pub age_seconds: f64,
    pub allowed: bool,
    /// Why the decision could not be made, or why the action was refused.
    /// Distinct fields on purpose: "we could not decide" and "we decided and
    /// then refused" are different states and only one of them is a bug.
    pub blocked: Option<&'static str>,
    pub refused: Option<&'static str>,
    pub commit_fraction: f64,
    pub held_fraction: f64,
    pub scores: Vec<policy::Score>,
}

impl T0Decision {
    /// The action a caller should actually take. Refusal collapses to doing
    /// nothing rather than to the next-best action: a policy whose second
    /// choice runs whenever its first is refused is a policy that routes
    /// around its own safety layer one step at a time.
    pub fn effective_action(&self) -> Action {
        if self.allowed {
            self.action
        } else if self.held_is_open() {
            Action::Hold
        } else {
            Action::Ignore
        }
    }

    /// Whether the decision was made about an open position, which is what
    /// decides which flavour of "do nothing" a refusal collapses to.
    fn held_is_open(&self) -> bool {
        self.held_fraction > 0.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Inputs<'a> {
    pub position: Position,
    pub survival: Survival,
    pub min_edge: f64,
    pub max_add_fraction: f64,
    pub live: bool,
    /// Distributions for REENTER and REPLACE. Absent for most decisions,
    /// which is why they are here rather than inside `Position`: a launch
    /// nobody has exited has no re-entry candidate.
    pub alternatives: policy::Alternatives<'a>,
}

/// One decision for one token, from state and a forward view.
pub fn decide(
    state: &TokenState,
    now: f64,
    inputs: &Inputs<'_>,
    limits: &Limits,
) -> T0Decision {
    let age_seconds = state.age_seconds(now);
    let band = policy::age_band(age_seconds);

    // A curve that cannot trade is not a decision with a bad answer, it is a
    // decision that cannot be made. Reported before pricing, because pricing
    // it would produce a number against reserves that do not exist.
    if !state.reserves.tradeable() {
        return T0Decision {
            action: Action::Hold,
            q: 0.0,
            age_band: band,
            age_seconds,
            allowed: false,
            blocked: Some("curve is not tradeable"),
            refused: None,
            commit_fraction: 0.0,
            held_fraction: inputs.position.held_fraction,
            scores: Vec::new(),
        };
    }

    let scored: PolicyDecision = policy::score_with(
        &inputs.position,
        &inputs.survival,
        inputs.min_edge,
        inputs.max_add_fraction,
        &inputs.alternatives,
    );
    if let Some(reason) = scored.blocked {
        return T0Decision {
            action: Action::Hold,
            q: 0.0,
            age_band: band,
            age_seconds,
            allowed: false,
            blocked: Some(reason),
            refused: None,
            commit_fraction: 0.0,
            held_fraction: inputs.position.held_fraction,
            scores: scored.scores,
        };
    }

    let commit_fraction = match scored.action {
        Action::Probe => inputs.position.probe_fraction.unwrap_or(0.0),
        Action::Add => inputs.position.add_fraction.unwrap_or(0.0),
        _ => 0.0,
    };

    let verdict = safety::check(
        &Intent {
            action: scored.action,
            held_fraction: inputs.position.held_fraction,
            commit_fraction,
            exit_capacity_ratio: inputs.position.exit_capacity_ratio,
            live: inputs.live,
        },
        limits,
    );

    T0Decision {
        action: scored.action,
        q: scored.q,
        age_band: band,
        age_seconds,
        allowed: matches!(verdict, Verdict::Allowed),
        blocked: None,
        refused: verdict.reason(),
        commit_fraction,
        held_fraction: inputs.position.held_fraction,
        scores: scored.scores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Buyer, Reserves};

    fn tradeable_state() -> TokenState {
        let mut state = TokenState::new(0.0);
        state.update_reserves(
            Reserves {
                virtual_sol: 30_000_000_000,
                virtual_token: 1_000_000_000_000,
                real_sol: 0,
                real_token: 0,
                measured: false,
            },
            0.0,
        );
        state.record_buy(Buyer {
            skill: Some(0.8),
            notional_sol: 2.0,
            at_seconds: 0.05,
            is_creator_linked: false,
        });
        state
    }

    fn inputs() -> Inputs<'static> {
        Inputs {
            position: Position {
                held_fraction: 0.0,
                current_multiple: 1.0,
                exit_cost: 0.02,
                entry_cost: 0.02,
                exit_capacity_ratio: Some(0.9),
                escape_probability: Some(0.9),
                alternative_growth_per_second: None,
                expected_remaining_seconds: None,
                add_fraction: None,
                add_capacity_fraction: None,
                probe_fraction: Some(0.02),
            },
            alternatives: policy::Alternatives::default(),
            survival: Survival {
                levels: [0.45, 0.3, 0.2, 0.12, 0.05, 0.0, 0.0, 0.0],
                p_rug_30s: 0.0,
                p_rug_5m: 0.0,
                expected_feasible_multiple: 0.0,
            },
            min_edge: 1e-4,
            max_add_fraction: 0.05,
            live: false,
        }
    }

    #[test]
    fn a_strong_flash_launch_probes_and_is_allowed() {
        let decision = decide(&tradeable_state(), 0.1, &inputs(), &Limits::default());
        assert_eq!(decision.action, Action::Probe);
        assert_eq!(decision.age_band, "flash");
        assert!(decision.allowed);
        assert!(decision.refused.is_none());
        assert!((decision.commit_fraction - 0.02).abs() < 1e-12);
    }

    #[test]
    fn an_untradeable_curve_blocks_before_pricing() {
        let state = TokenState::new(0.0);
        let decision = decide(&state, 0.1, &inputs(), &Limits::default());
        assert_eq!(decision.blocked, Some("curve is not tradeable"));
        assert!(decision.scores.is_empty());
    }

    #[test]
    fn unmeasured_escape_blocks_rather_than_being_refused() {
        // "Could not decide" and "decided then refused" are different states
        // and only one of them is a bug.
        let mut blind = inputs();
        blind.position.escape_probability = None;
        let decision = decide(&tradeable_state(), 0.1, &blind, &Limits::default());
        assert!(decision.blocked.is_some());
        assert!(decision.refused.is_none());
    }

    #[test]
    fn a_locked_live_desk_decides_and_then_refuses() {
        let mut live = inputs();
        live.live = true;
        let decision = decide(&tradeable_state(), 0.1, &live, &Limits::default());
        assert_eq!(decision.action, Action::Probe);
        assert!(decision.blocked.is_none());
        assert_eq!(decision.refused, Some("live submission is locked"));
        assert!(!decision.allowed);
        // And the effective action is to do nothing, not to fall through to
        // whatever scored second. On a flat book "nothing" is IGNORE, which
        // is what gets recorded so the launch can be scored later against
        // what it went on to do.
        assert_eq!(decision.effective_action(), Action::Ignore);
    }

    #[test]
    fn a_refused_entry_does_not_fall_through_to_the_runner_up() {
        let mut oversized = inputs();
        oversized.position.probe_fraction = Some(0.9);
        let decision = decide(&tradeable_state(), 0.1, &oversized, &Limits::default());
        assert!(!decision.allowed);
        assert_eq!(decision.effective_action(), Action::Ignore);

        // An open position refused an ADD holds rather than ignoring: it has
        // exposure, and "ignore" would misdescribe what is happening to it.
        let mut open = oversized;
        open.position.held_fraction = 0.2;
        open.position.probe_fraction = None;
        open.position.add_fraction = Some(0.9);
        let held = decide(&tradeable_state(), 0.1, &open, &Limits::default());
        if !held.allowed {
            assert_eq!(held.effective_action(), Action::Hold);
        }
    }

    #[test]
    fn the_age_band_travels_with_the_decision() {
        let state = tradeable_state();
        assert_eq!(
            decide(&state, 0.2, &inputs(), &Limits::default()).age_band,
            "flash"
        );
        assert_eq!(
            decide(&state, 2.0, &inputs(), &Limits::default()).age_band,
            "early"
        );
        assert_eq!(
            decide(&state, 30.0, &inputs(), &Limits::default()).age_band,
            "forming"
        );
        assert_eq!(
            decide(&state, 600.0, &inputs(), &Limits::default()).age_band,
            "mature"
        );
    }

    #[test]
    fn an_exit_is_never_refused_by_safety() {
        let mut exiting = inputs();
        exiting.position.held_fraction = 0.3;
        exiting.position.current_multiple = 2.0;
        exiting.position.probe_fraction = None;
        exiting.survival = Survival {
            levels: [0.2, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            p_rug_30s: 0.5,
            p_rug_5m: 0.6,
            expected_feasible_multiple: 0.0,
        };
        exiting.live = true;
        let decision = decide(&tradeable_state(), 5.0, &exiting, &Limits::default());
        assert!(decision.action.bank_fraction() > 0.0);
        assert!(
            decision.allowed,
            "safety refused an exit: {:?}",
            decision.refused
        );
    }
}
