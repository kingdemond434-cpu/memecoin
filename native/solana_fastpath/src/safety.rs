//! Invariants no policy output may bypass.
//!
//! The action-value engine owns what the desk *should* do. This owns what it
//! is *allowed* to do, and the separation matters: a policy is a model of the
//! world and models are wrong, whereas these are arithmetic facts about the
//! account. A size larger than the book is not a bad trade, it is an
//! impossible one, and no expected-growth argument makes it possible.
//!
//! So these are checked after the policy has chosen and before anything is
//! built. They cannot be traded off, weighted, or improved by evidence -- the
//! moment a safety limit is something the objective can outbid, it has
//! stopped being a limit.
//!
//! Deliberately few. A long list of soft guards is a policy wearing a
//! safety's clothes, and it dilutes the ones that matter: every check here is
//! one where the correct response is refusing rather than resizing.

use crate::policy::Action;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// Largest share of the book one position may hold, after the action.
    pub max_position_fraction: f64,
    /// Largest share of the book one action may commit at once.
    pub max_single_commit_fraction: f64,
    /// Below this, a fill is dust: the fees exceed anything it can return.
    pub min_commit_fraction: f64,
    /// Refuse to buy into a book we could not sell a meaningful slice of.
    pub min_exit_capacity: f64,
    /// Live capital requires the acknowledgement. Dry run does not.
    pub live_unlocked: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_position_fraction: 0.25,
            max_single_commit_fraction: 0.05,
            min_commit_fraction: 0.0005,
            min_exit_capacity: 0.10,
            live_unlocked: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Intent {
    pub action: Action,
    pub held_fraction: f64,
    /// Share of the book this action commits. Zero for exits and holds.
    pub commit_fraction: f64,
    pub exit_capacity_ratio: Option<f64>,
    pub live: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    Allowed,
    Refused(&'static str),
}

impl Verdict {
    pub fn allowed(self) -> bool {
        matches!(self, Verdict::Allowed)
    }

    pub fn reason(self) -> Option<&'static str> {
        match self {
            Verdict::Refused(reason) => Some(reason),
            Verdict::Allowed => None,
        }
    }
}

/// Check one intended action against the limits.
///
/// Selling is never refused. Every check here exists to stop capital going
/// out, and a guard that can block an exit is a guard that can trap a
/// position in exactly the conditions it was written to protect against.
pub fn check(intent: &Intent, limits: &Limits) -> Verdict {
    if intent.commit_fraction <= 0.0 || intent.action.bank_fraction() > 0.0 {
        // Exits, banks and holds commit nothing. Nothing to refuse.
        return Verdict::Allowed;
    }
    if intent.live && !limits.live_unlocked {
        return Verdict::Refused("live submission is locked");
    }
    if !intent.commit_fraction.is_finite() || !intent.held_fraction.is_finite() {
        return Verdict::Refused("non-finite size");
    }
    if intent.commit_fraction < limits.min_commit_fraction {
        return Verdict::Refused("commit is dust; fees exceed what it can return");
    }
    if intent.commit_fraction > limits.max_single_commit_fraction {
        return Verdict::Refused("commit exceeds the single-action limit");
    }
    if intent.held_fraction + intent.commit_fraction > limits.max_position_fraction {
        return Verdict::Refused("position would exceed the per-token limit");
    }
    match intent.exit_capacity_ratio {
        // Unmeasured is refused, not assumed liquid. An entry into a book we
        // cannot measure our way out of is an entry we cannot size.
        None => Verdict::Refused("exit capacity not measured"),
        Some(capacity) if capacity < limits.min_exit_capacity => {
            Verdict::Refused("exit capacity below the floor")
        }
        Some(_) => Verdict::Allowed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intent() -> Intent {
        Intent {
            action: Action::Probe,
            held_fraction: 0.0,
            commit_fraction: 0.02,
            exit_capacity_ratio: Some(0.9),
            live: false,
        }
    }

    #[test]
    fn a_reasonable_probe_is_allowed() {
        assert!(check(&intent(), &Limits::default()).allowed());
    }

    #[test]
    fn selling_is_never_refused() {
        // A guard that can block an exit traps a position in exactly the
        // conditions it was written to protect against.
        for action in [Action::Exit, Action::Bank50, Action::Bank75] {
            let selling = Intent {
                action,
                held_fraction: 0.9,
                commit_fraction: 0.0,
                exit_capacity_ratio: None,
                live: true,
            };
            assert!(check(&selling, &Limits::default()).allowed());
        }
    }

    #[test]
    fn live_capital_needs_the_acknowledgement() {
        let mut live = intent();
        live.live = true;
        assert_eq!(
            check(&live, &Limits::default()).reason(),
            Some("live submission is locked")
        );
        let unlocked = Limits {
            live_unlocked: true,
            ..Limits::default()
        };
        assert!(check(&live, &unlocked).allowed());
    }

    #[test]
    fn dust_is_refused_rather_than_rounded_up() {
        let mut dust = intent();
        dust.commit_fraction = 1e-9;
        assert!(check(&dust, &Limits::default()).reason().is_some());
    }

    #[test]
    fn the_single_commit_and_position_limits_are_separate() {
        let mut oversized = intent();
        oversized.commit_fraction = 0.2;
        assert_eq!(
            check(&oversized, &Limits::default()).reason(),
            Some("commit exceeds the single-action limit")
        );

        // Within the single-action limit, but the position would breach.
        let mut topping_up = intent();
        topping_up.action = Action::Add;
        topping_up.held_fraction = 0.24;
        topping_up.commit_fraction = 0.02;
        assert_eq!(
            check(&topping_up, &Limits::default()).reason(),
            Some("position would exceed the per-token limit")
        );
    }

    #[test]
    fn unmeasured_capacity_is_refused_not_assumed_liquid() {
        let mut unknown = intent();
        unknown.exit_capacity_ratio = None;
        assert_eq!(
            check(&unknown, &Limits::default()).reason(),
            Some("exit capacity not measured")
        );
    }

    #[test]
    fn a_thin_book_is_refused() {
        let mut thin = intent();
        thin.exit_capacity_ratio = Some(0.01);
        assert_eq!(
            check(&thin, &Limits::default()).reason(),
            Some("exit capacity below the floor")
        );
    }

    #[test]
    fn nonfinite_sizes_are_refused() {
        let mut broken = intent();
        broken.commit_fraction = f64::NAN;
        assert!(check(&broken, &Limits::default()).reason().is_some());
        broken.commit_fraction = f64::INFINITY;
        assert!(check(&broken, &Limits::default()).reason().is_some());
    }
}
