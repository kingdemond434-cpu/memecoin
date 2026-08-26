//! The decision itself: survival curve -> executable bins -> one Q per action.
//!
//! This is `src/strategies/action_value.py` in Rust, and it is deliberately a
//! mirror rather than an improvement. Two implementations of one objective
//! that disagree are worse than either alone, because the disagreement shows
//! up as trades nobody can explain -- so the parity test drives both from the
//! same inputs and requires the same answer, and any change here has to be
//! made there too.
//!
//! What it buys is that the whole decision -- bins, capture, every action's
//! value, the comparison -- happens in one pass with no Python objects, on
//! the path where an event arrives and something has to be done about it.
//!
//! The economics are the Python module's, restated so this file stands alone:
//!
//! Holding is the baseline and scores exactly zero, so doing nothing wins
//! ties and a policy that cannot tell two actions apart does not churn the
//! book. Every action prices the same forward distribution, because two
//! components cannot disagree about a number they both read from one place.
//! Banking is charged both legs -- the exit cost and the surrendered upside --
//! since a free bank always beats holding and a policy that banks every tick
//! turns a runner into a fee schedule. And upside that cannot be sold, or
//! that we are unlikely to escape with, is not upside: capacity and escape
//! multiply every positive outcome, with no permissive default for either.

use crate::state::TokenState;

/// Cumulative survival levels, ascending. Mirrors `SURVIVAL_LEVELS`.
pub const SURVIVAL_MULTIPLES: [f64; 8] = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 500.0];

/// What a position can do. Mirrors `Action`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Ignore,
    Probe,
    Hold,
    Add,
    Bank10,
    Bank25,
    Bank50,
    Bank75,
    Exit,
    Reenter,
    Replace,
}

impl Action {
    pub fn bank_fraction(self) -> f64 {
        match self {
            Action::Bank10 => 0.10,
            Action::Bank25 => 0.25,
            Action::Bank50 => 0.50,
            Action::Bank75 => 0.75,
            Action::Exit | Action::Replace => 1.0,
            _ => 0.0,
        }
    }

    pub fn is_entry(self) -> bool {
        matches!(self, Action::Ignore | Action::Probe)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Action::Ignore => "ignore",
            Action::Probe => "probe",
            Action::Hold => "hold",
            Action::Add => "add",
            Action::Bank10 => "bank_10",
            Action::Bank25 => "bank_25",
            Action::Bank50 => "bank_50",
            Action::Bank75 => "bank_75",
            Action::Exit => "exit",
            Action::Reenter => "reenter",
            Action::Replace => "replace",
        }
    }
}

/// One disjoint outcome: probability and gross return.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bin {
    pub probability: f64,
    pub gross: f64,
}

/// The forward view, as the predictor reports it.
#[derive(Debug, Clone, Copy)]
pub struct Survival {
    /// P(>= each multiple), in the order of `SURVIVAL_MULTIPLES`.
    pub levels: [f64; 8],
    pub p_rug_30s: f64,
    pub p_rug_5m: f64,
    /// Zero when unmeasured, in which case no ceiling is applied.
    pub expected_feasible_multiple: f64,
}

/// Everything an action needs to be priced, at one instant.
#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub held_fraction: f64,
    pub current_multiple: f64,
    pub exit_cost: f64,
    pub entry_cost: f64,
    /// `None` means unmeasured, and unmeasured blocks the whole decision.
    /// Reading unknown escape as certain escape is the single most flattering
    /// assumption available, and it flatters hardest on exactly the tokens
    /// where escape is hardest to measure.
    pub exit_capacity_ratio: Option<f64>,
    pub escape_probability: Option<f64>,
    pub alternative_growth_per_second: Option<f64>,
    pub expected_remaining_seconds: Option<f64>,
    pub add_fraction: Option<f64>,
    pub add_capacity_fraction: Option<f64>,
    pub probe_fraction: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub action: Action,
    pub q: f64,
    pub feasible: bool,
}

#[derive(Debug, Clone)]
pub struct Decision {
    pub blocked: Option<&'static str>,
    pub action: Action,
    pub q: f64,
    pub scores: Vec<Score>,
}

/// Nested probabilities cannot rise as the multiple rises. Heads are trained
/// independently, so they can. An untrained tail head reads zero and this
/// collapses the mass into the highest rung the model can actually see --
/// which is correct: extending the curve must not manufacture conviction.
pub fn enforce_monotonic(levels: &mut [f64; 8]) {
    let mut ceiling = 1.0;
    for level in levels.iter_mut() {
        let clamped = level.clamp(0.0, 1.0);
        let value = clamped.min(ceiling);
        *level = value;
        ceiling = value;
    }
}

/// Disjoint outcomes from the cumulative curve. Mirrors `probability_bins`.
///
/// Every bucket pays its own LOWER bound rather than a midpoint: paying the
/// midpoint of a bucket the model was never asked about is inventing the
/// shape of the distribution inside it, while the lower bound cannot
/// overstate what the bucket contains.
pub fn probability_bins(survival: &Survival) -> Vec<Bin> {
    let mut levels = survival.levels;
    enforce_monotonic(&mut levels);

    let mut raw: Vec<Bin> = Vec::with_capacity(SURVIVAL_MULTIPLES.len() + 2);
    raw.push(Bin {
        probability: (1.0 - levels[0]).max(0.0),
        gross: -0.35,
    });
    for index in 0..SURVIVAL_MULTIPLES.len() {
        let higher = if index + 1 < levels.len() {
            levels[index + 1]
        } else {
            0.0
        };
        raw.push(Bin {
            probability: (levels[index] - higher).max(0.0),
            gross: SURVIVAL_MULTIPLES[index] - 1.0,
        });
    }

    if survival.expected_feasible_multiple > 0.0 {
        let ceiling = (survival.expected_feasible_multiple - 1.0)
            .clamp(-0.98, SURVIVAL_MULTIPLES[SURVIVAL_MULTIPLES.len() - 1] - 1.0);
        for bin in raw.iter_mut() {
            if bin.gross > 0.0 {
                bin.gross = bin.gross.min(ceiling);
            }
        }
    }

    let p_rug = survival.p_rug_30s.max(survival.p_rug_5m).clamp(0.0, 1.0);
    let mut bins: Vec<Bin> = raw
        .into_iter()
        .map(|bin| Bin {
            probability: bin.probability * (1.0 - p_rug),
            gross: bin.gross,
        })
        .collect();
    bins.push(Bin {
        probability: p_rug,
        gross: -0.98,
    });

    let total: f64 = bins.iter().map(|bin| bin.probability).sum();
    if total <= 0.0 {
        return Vec::new();
    }
    for bin in bins.iter_mut() {
        bin.probability /= total;
    }
    bins
}

fn capture(position: &Position, gross: f64) -> f64 {
    if gross <= 0.0 {
        return gross;
    }
    let capacity = position.exit_capacity_ratio.unwrap_or(0.0).clamp(0.0, 1.0);
    let escape = position.escape_probability.unwrap_or(0.0).clamp(0.0, 1.0);
    gross * capacity * escape
}

/// E[log W] over the bins, or -inf if any outcome wipes the book.
fn expected_log(bins: &[Bin], wealth: impl Fn(f64) -> f64) -> f64 {
    let mut total = 0.0;
    for bin in bins {
        let value = wealth(bin.gross);
        if value <= 0.0 {
            return f64::NEG_INFINITY;
        }
        total += bin.probability * value.ln();
    }
    total
}

fn hold_value(position: &Position, bins: &[Bin]) -> f64 {
    let held = position.held_fraction;
    let multiple = position.current_multiple.max(0.0);
    let cash = 1.0 - held;
    let position_now = held * multiple;
    expected_log(bins, |gross| {
        cash + position_now * (1.0 + capture(position, gross))
    })
}

fn bank_value(position: &Position, bins: &[Bin], fraction: f64) -> f64 {
    let held = position.held_fraction;
    let multiple = position.current_multiple.max(0.0);
    let capacity = position.exit_capacity_ratio.unwrap_or(1.0).clamp(0.0, 1.0);
    // Only the part the venue can absorb is actually sold.
    let sold = held * fraction * capacity;
    let remaining = held - sold;
    let proceeds = sold * multiple * (1.0 - position.exit_cost);
    let cash = (1.0 - held) + proceeds;
    let position_now = remaining * multiple;
    expected_log(bins, |gross| {
        cash + position_now * (1.0 + capture(position, gross))
    })
}

fn redeploy_bonus(position: &Position) -> f64 {
    match (
        position.alternative_growth_per_second,
        position.expected_remaining_seconds,
    ) {
        (Some(rate), Some(seconds)) if rate > 0.0 && seconds > 0.0 => {
            let freed = position.held_fraction * position.current_multiple.max(0.0);
            rate * seconds * freed
        }
        _ => 0.0,
    }
}

fn add_value(position: &Position, bins: &[Bin], max_add: f64) -> f64 {
    let added = match position.add_fraction {
        Some(value) if value > 0.0 => value,
        _ => return f64::NEG_INFINITY,
    };
    let capacity_ceiling = position.add_capacity_fraction.unwrap_or(max_add);
    if added > max_add.min(capacity_ceiling) {
        return f64::NEG_INFINITY;
    }
    let held = position.held_fraction;
    let multiple = position.current_multiple.max(0.0);
    let cash = 1.0 - held - added;
    if cash < 0.0 {
        return f64::NEG_INFINITY;
    }
    let position_now = held * multiple + added;
    expected_log(bins, |gross| {
        cash + position_now * (1.0 + capture(position, gross)) - added * position.entry_cost
    })
}

fn probe_value(position: &Position, bins: &[Bin]) -> f64 {
    if position.held_fraction > 0.0 {
        return f64::NEG_INFINITY;
    }
    let added = match position.probe_fraction {
        Some(value) if value > 0.0 => value,
        _ => return f64::NEG_INFINITY,
    };
    let cash = 1.0 - added;
    if cash < 0.0 {
        return f64::NEG_INFINITY;
    }
    expected_log(bins, |gross| {
        cash + added * (1.0 + capture(position, gross)) - added * position.entry_cost
    })
}

/// Score every action against holding. Mirrors `ActionValuePolicy.score`.
pub fn score(position: &Position, survival: &Survival, min_edge: f64, max_add: f64) -> Decision {
    let bins = probability_bins(survival);
    if bins.is_empty() {
        return blocked("no forward distribution");
    }
    if position.exit_capacity_ratio.is_none() {
        return blocked("exit capacity not measured");
    }
    if position.escape_probability.is_none() {
        return blocked("escape probability not measured");
    }
    if !(0.0..=1.0).contains(&position.held_fraction) {
        return blocked("held fraction out of range");
    }
    if position.current_multiple < 0.0 {
        return blocked("negative multiple");
    }

    let baseline = hold_value(position, &bins);
    if !baseline.is_finite() {
        return blocked("holding has no finite value; the state is not priceable");
    }

    let flat = position.held_fraction <= 0.0;
    let mut scores: Vec<Score> = Vec::with_capacity(11);
    let mut push = |action: Action, value: f64| {
        let q = value - baseline;
        scores.push(Score {
            action,
            q,
            feasible: q.is_finite(),
        });
    };

    push(Action::Hold, baseline);
    for action in [
        Action::Bank10,
        Action::Bank25,
        Action::Bank50,
        Action::Bank75,
    ] {
        push(action, bank_value(position, &bins, action.bank_fraction()));
    }
    push(
        Action::Exit,
        bank_value(position, &bins, 1.0) + redeploy_bonus(position),
    );
    push(Action::Add, add_value(position, &bins, max_add));
    push(Action::Probe, probe_value(position, &bins));
    push(
        Action::Ignore,
        if flat { baseline } else { f64::NEG_INFINITY },
    );

    let best = scores
        .iter()
        .filter(|score| score.feasible)
        .copied()
        .fold(None::<Score>, |best, score| match best {
            Some(current) if current.q >= score.q => Some(current),
            _ => Some(score),
        });

    // Doing nothing wins ties and wins anything inside the noise margin.
    // Which "nothing" it is depends on whether anything is held: recording
    // the difference is what lets a rejected launch be scored against what it
    // went on to do.
    let idle = if flat { Action::Ignore } else { Action::Hold };
    match best {
        Some(score) if score.q > min_edge => Decision {
            blocked: None,
            action: score.action,
            q: score.q,
            scores,
        },
        _ => Decision {
            blocked: None,
            action: idle,
            q: 0.0,
            scores,
        },
    }
}

fn blocked(reason: &'static str) -> Decision {
    Decision {
        blocked: Some(reason),
        action: Action::Hold,
        q: 0.0,
        scores: Vec::new(),
    }
}

/// Seconds of launch age each age band covers. Mirrors `AGE_BANDS`.
pub fn age_band(age_seconds: f64) -> &'static str {
    let age = age_seconds.max(0.0);
    if age < 0.5 {
        "flash"
    } else if age < 5.0 {
        "early"
    } else if age < 60.0 {
        "forming"
    } else {
        "mature"
    }
}

/// A compact view of a token's state for the decision path.
pub fn summarise(state: &TokenState, now: f64) -> [f64; 8] {
    let (skill, scored) = state.cohort_skill();
    [
        state.age_seconds(now),
        state.observed_buyers() as f64,
        skill.unwrap_or(-1.0),
        scored as f64,
        state.creator_linked_share().unwrap_or(-1.0),
        state.buy_velocity().unwrap_or(-1.0),
        state.net_flow_sol(),
        if state.creator_sold { 1.0 } else { 0.0 },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn survival(levels: [f64; 8]) -> Survival {
        Survival {
            levels,
            p_rug_30s: 0.0,
            p_rug_5m: 0.0,
            expected_feasible_multiple: 0.0,
        }
    }

    fn position() -> Position {
        Position {
            held_fraction: 0.3,
            current_multiple: 2.0,
            exit_cost: 0.02,
            entry_cost: 0.02,
            exit_capacity_ratio: Some(1.0),
            escape_probability: Some(1.0),
            alternative_growth_per_second: None,
            expected_remaining_seconds: None,
            add_fraction: None,
            add_capacity_fraction: None,
            probe_fraction: None,
        }
    }

    #[test]
    fn bins_sum_to_one_and_pay_lower_bounds() {
        let bins = probability_bins(&survival([0.5, 0.3, 0.2, 0.12, 0.06, 0.03, 0.01, 0.004]));
        let total: f64 = bins.iter().map(|bin| bin.probability).sum();
        assert!((total - 1.0).abs() < 1e-9);
        // The top bucket pays 499, not a midpoint above it.
        assert!(bins.iter().any(|bin| (bin.gross - 499.0).abs() < 1e-9));
    }

    #[test]
    fn an_untrained_tail_collapses_rather_than_inventing_conviction() {
        let bins = probability_bins(&survival([0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]));
        let tail: f64 = bins
            .iter()
            .filter(|bin| bin.gross > 49.0)
            .map(|bin| bin.probability)
            .sum();
        assert!(tail.abs() < 1e-12);
    }

    #[test]
    fn monotonicity_is_enforced_not_assumed() {
        let mut levels = [0.4, 0.7, 0.2, 0.9, 0.05, 0.0, 0.0, 0.0];
        enforce_monotonic(&mut levels);
        for pair in levels.windows(2) {
            assert!(pair[0] >= pair[1]);
        }
    }

    #[test]
    fn unmeasured_capacity_or_escape_blocks_the_whole_decision() {
        let mut blocked_position = position();
        blocked_position.exit_capacity_ratio = None;
        let decision = score(&blocked_position, &survival([0.5; 8]), 1e-4, 0.05);
        assert!(decision.blocked.is_some());

        let mut other = position();
        other.escape_probability = None;
        assert!(score(&other, &survival([0.5; 8]), 1e-4, 0.05)
            .blocked
            .is_some());
    }

    #[test]
    fn add_respects_the_live_liquidity_capacity_ceiling() {
        let mut capped = position();
        capped.add_fraction = Some(0.04);
        capped.add_capacity_fraction = Some(0.02);
        let decision = score(
            &capped,
            &survival([0.9, 0.7, 0.5, 0.3, 0.1, 0.03, 0.01, 0.004]),
            1e-4,
            0.05,
        );
        let add = decision
            .scores
            .iter()
            .find(|score| score.action == Action::Add)
            .expect("ADD score exists");
        assert!(!add.feasible);
    }

    #[test]
    fn holding_is_the_baseline_and_scores_zero() {
        let decision = score(
            &position(),
            &survival([0.5, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0]),
            1e-4,
            0.05,
        );
        let hold = decision
            .scores
            .iter()
            .find(|score| score.action == Action::Hold)
            .unwrap();
        assert!(hold.q.abs() < 1e-12);
    }

    #[test]
    fn doing_nothing_is_ignore_when_flat_and_hold_when_open() {
        // Isolated with an unreachable margin, so nothing clears and the ONLY
        // thing under test is which "nothing" gets recorded. Picking a
        // distribution where both books happen to sit still would test the
        // distribution instead, and there is no reason to expect one exists.
        let unreachable_margin = 1e9;
        let any = survival([0.5, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0]);

        let mut flat = position();
        flat.held_fraction = 0.0;
        flat.probe_fraction = Some(0.02);
        let flat_decision = score(&flat, &any, unreachable_margin, 0.05);
        assert_eq!(flat_decision.action, Action::Ignore);
        assert_eq!(flat_decision.q, 0.0);

        let open_decision = score(&position(), &any, unreachable_margin, 0.05);
        assert_eq!(open_decision.action, Action::Hold);
        assert_eq!(open_decision.q, 0.0);
    }

    #[test]
    fn a_genuinely_weak_flat_book_declines_to_probe() {
        let mut flat = position();
        flat.held_fraction = 0.0;
        flat.current_multiple = 1.0;
        flat.probe_fraction = Some(0.02);
        // 80% below 2x at -0.35 outweighs what this thin tail pays.
        // [0.2, 0.05, ...] would NOT be weak -- it clears zero -- and
        // asserting IGNORE against it would have been asserting a bug.
        let weak = survival([0.2, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(score(&flat, &weak, 1e-4, 0.05).action, Action::Ignore);
    }

    #[test]
    fn a_weak_forward_view_exits_an_open_position() {
        let weak = survival([0.2, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(score(&position(), &weak, 1e-4, 0.05).action, Action::Exit);
    }

    #[test]
    fn a_worthwhile_flat_book_probes() {
        let mut flat = position();
        flat.held_fraction = 0.0;
        flat.current_multiple = 1.0;
        flat.probe_fraction = Some(0.02);
        let strong = survival([0.45, 0.3, 0.2, 0.12, 0.05, 0.0, 0.0, 0.0]);
        assert_eq!(score(&flat, &strong, 1e-4, 0.05).action, Action::Probe);
    }

    #[test]
    fn probing_is_unavailable_once_a_position_is_open() {
        let mut open = position();
        open.probe_fraction = Some(0.02);
        let decision = score(&open, &survival([0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]), 1e-4, 0.05);
        let probe = decision
            .scores
            .iter()
            .find(|score| score.action == Action::Probe)
            .unwrap();
        assert!(!probe.feasible);
    }

    #[test]
    fn banking_is_never_free() {
        // With no exit cost advantage and a healthy forward view, banking
        // must not beat holding: a free bank always would.
        let strong = survival([0.9, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0]);
        let decision = score(&position(), &strong, 1e-4, 0.05);
        let bank = decision
            .scores
            .iter()
            .find(|score| score.action == Action::Bank50)
            .unwrap();
        assert!(bank.q < 0.0, "banking scored {} against holding", bank.q);
    }

    #[test]
    fn an_add_beyond_the_ceiling_is_infeasible() {
        let mut adding = position();
        adding.add_fraction = Some(0.5);
        let decision = score(&adding, &survival([0.6, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]), 1e-4, 0.05);
        let add = decision
            .scores
            .iter()
            .find(|score| score.action == Action::Add)
            .unwrap();
        assert!(!add.feasible);
    }

    #[test]
    fn a_certain_rug_makes_exiting_the_answer() {
        let doomed = Survival {
            levels: [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            p_rug_30s: 0.95,
            p_rug_5m: 0.99,
            expected_feasible_multiple: 0.0,
        };
        let decision = score(&position(), &doomed, 1e-4, 0.05);
        assert!(matches!(
            decision.action,
            Action::Exit | Action::Bank75 | Action::Bank50
        ));
    }

    #[test]
    fn the_age_bands_partition_the_timeline() {
        assert_eq!(age_band(0.0), "flash");
        assert_eq!(age_band(0.499), "flash");
        assert_eq!(age_band(0.5), "early");
        assert_eq!(age_band(59.9), "forming");
        assert_eq!(age_band(60.0), "mature");
    }

    #[test]
    fn unmeasured_summary_fields_are_negative_not_zero() {
        let state = TokenState::new(0.0);
        let summary = summarise(&state, 1.0);
        // skill, linked share and velocity are all unmeasured here.
        assert_eq!(summary[2], -1.0);
        assert_eq!(summary[4], -1.0);
        assert_eq!(summary[5], -1.0);
    }
}
