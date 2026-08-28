//! Nanosecond stage timing through the hot path.
//!
//! "The bot is slow" is not an actionable statement. Optimising the stage that
//! feels slow, rather than the stage the percentiles indict, is how weeks go
//! into a decode that was never the bottleneck. Every stage is timed
//! separately so the next engineering hour goes where the distribution says.
//!
//! Percentiles rather than means, because the mean of a latency distribution
//! with a fat tail is a number no individual event ever experienced, and it is
//! the tail that loses the race.

use std::collections::HashMap;
use std::time::Instant;

/// The stages a launch event passes through before a transaction is signed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    Receive,
    Decode,
    Features,
    Predict,
    Size,
    Build,
}

impl Stage {
    pub fn name(&self) -> &'static str {
        match self {
            Stage::Receive => "receive",
            Stage::Decode => "decode",
            Stage::Features => "features",
            Stage::Predict => "predict",
            Stage::Size => "size",
            Stage::Build => "build",
        }
    }
}

/// Fixed-capacity ring of durations, in nanoseconds.
///
/// Bounded on purpose: an unbounded latency log on a 4 GB node is a slow
/// memory leak that only shows up under the load it is meant to measure.
#[derive(Debug)]
pub struct Samples {
    values: Vec<u64>,
    capacity: usize,
    next: usize,
    pub observed: u64,
}

impl Samples {
    pub fn new(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            capacity: capacity.max(1),
            next: 0,
            observed: 0,
        }
    }

    pub fn record(&mut self, nanos: u64) {
        self.observed += 1;
        if self.values.len() < self.capacity {
            self.values.push(nanos);
        } else {
            self.values[self.next] = nanos;
            self.next = (self.next + 1) % self.capacity;
        }
    }

    /// Nearest-rank percentile, or `None` when nothing has been recorded.
    ///
    /// `None` rather than zero: a stage that has never run and a stage that
    /// runs instantly are different facts, and only one of them means the hot
    /// path is fast.
    pub fn percentile(&self, p: f64) -> Option<u64> {
        if self.values.is_empty() {
            return None;
        }
        let mut sorted = self.values.clone();
        sorted.sort_unstable();
        let rank = ((p.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize).max(1);
        sorted.get(rank - 1).copied()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[derive(Debug)]
pub struct Telemetry {
    stages: HashMap<Stage, Samples>,
    capacity: usize,
}

impl Telemetry {
    pub fn new(capacity: usize) -> Self {
        Self {
            stages: HashMap::new(),
            capacity,
        }
    }

    pub fn record(&mut self, stage: Stage, nanos: u64) {
        self.stages
            .entry(stage)
            .or_insert_with(|| Samples::new(self.capacity))
            .record(nanos);
    }

    pub fn time<T>(&mut self, stage: Stage, work: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let result = work();
        self.record(stage, started.elapsed().as_nanos() as u64);
        result
    }

    pub fn percentile(&self, stage: Stage, p: f64) -> Option<u64> {
        self.stages
            .get(&stage)
            .and_then(|samples| samples.percentile(p))
    }

    /// (stage, p50, p90, p99, count) for every stage that has run.
    #[allow(clippy::type_complexity)]
    pub fn report(&self) -> Vec<(&'static str, Option<u64>, Option<u64>, Option<u64>, u64)> {
        let mut rows: Vec<_> = self
            .stages
            .iter()
            .map(|(stage, samples)| {
                (
                    stage.name(),
                    samples.percentile(0.50),
                    samples.percentile(0.90),
                    samples.percentile(0.99),
                    samples.observed,
                )
            })
            .collect();
        // Slowest p99 first: the stage that loses the race, not the noisiest.
        rows.sort_by_key(|row| std::cmp::Reverse(row.3.unwrap_or(0)));
        rows
    }

    /// The stage with the worst p99, which is the one worth optimising.
    pub fn bottleneck(&self) -> Option<(&'static str, u64)> {
        self.report()
            .into_iter()
            .find_map(|(name, _, _, p99, _)| p99.map(|value| (name, value)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unrun_stage_reports_none_not_zero() {
        let telemetry = Telemetry::new(16);
        // "Never ran" and "runs instantly" are different facts and only one
        // means the hot path is fast.
        assert_eq!(telemetry.percentile(Stage::Decode, 0.5), None);
        assert_eq!(telemetry.bottleneck(), None);
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let mut samples = Samples::new(100);
        for value in 1..=100u64 {
            samples.record(value);
        }
        assert_eq!(samples.percentile(0.50), Some(50));
        assert_eq!(samples.percentile(0.90), Some(90));
        assert_eq!(samples.percentile(0.99), Some(99));
        assert_eq!(samples.percentile(1.0), Some(100));
    }

    #[test]
    fn the_ring_is_bounded_and_keeps_the_newest() {
        let mut samples = Samples::new(4);
        for value in 1..=10u64 {
            samples.record(value);
        }
        // An unbounded latency log is a slow leak that appears under exactly
        // the load it exists to measure.
        assert_eq!(samples.len(), 4);
        assert_eq!(samples.observed, 10);
        assert_eq!(samples.percentile(1.0), Some(10));
    }

    #[test]
    fn the_bottleneck_is_the_worst_p99_not_the_loudest_stage() {
        let mut telemetry = Telemetry::new(64);
        for _ in 0..50 {
            telemetry.record(Stage::Decode, 1_000);
            telemetry.record(Stage::Predict, 50_000);
        }
        // Decode ran as often; predict is what loses the race.
        assert_eq!(telemetry.bottleneck(), Some(("predict", 50_000)));
    }

    #[test]
    fn timing_a_closure_records_the_stage() {
        let mut telemetry = Telemetry::new(8);
        let result = telemetry.time(Stage::Build, || 21 * 2);
        assert_eq!(result, 42);
        assert!(telemetry.percentile(Stage::Build, 0.5).is_some());
    }

    #[test]
    fn the_report_lists_every_stage_that_ran() {
        let mut telemetry = Telemetry::new(8);
        telemetry.record(Stage::Receive, 10);
        telemetry.record(Stage::Decode, 20);
        assert_eq!(telemetry.report().len(), 2);
    }
}
