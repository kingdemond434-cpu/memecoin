//! The L0 reject, run before anything expensive touches an event.
//!
//! Watching thousands of launches at once only works if the overwhelming
//! majority are dismissed before any feature is computed, any wallet is looked
//! up, or any Python object is allocated. This stage exists to be wrong in one
//! direction only: it may pass an event that later stages reject, and it must
//! never reject one they would have accepted.
//!
//! That asymmetry is why the checks here are all structural -- discriminators,
//! lengths, program ids. Nothing here judges whether a launch is *good*. A
//! filter that started making quality judgements at L0 would silently become
//! the model, without any of the validation a model has to pass.

/// Anchor discriminators the hot path recognises, matched by prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Create,
    Trade,
    Complete,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reject {
    TooShort,
    UnknownDiscriminator,
    ForeignProgram,
}

pub struct L0Filter {
    create: [u8; 8],
    trade: [u8; 8],
    complete: [u8; 8],
    program_id: [u8; 32],
    pub passed: u64,
    pub rejected: u64,
}

impl L0Filter {
    pub fn new(
        program_id: [u8; 32],
        create: [u8; 8],
        trade: [u8; 8],
        complete: [u8; 8],
    ) -> Self {
        Self { create, trade, complete, program_id, passed: 0, rejected: 0 }
    }

    /// Classify one program-data payload, or say why it was dropped.
    ///
    /// The program check comes first because it is a fixed 32-byte compare
    /// that eliminates every event from every other program on the chain,
    /// which is the overwhelming majority of them.
    pub fn classify(&mut self, program_id: &[u8], data: &[u8]) -> Result<EventKind, Reject> {
        if program_id != self.program_id {
            self.rejected += 1;
            return Err(Reject::ForeignProgram);
        }
        if data.len() < 8 {
            self.rejected += 1;
            return Err(Reject::TooShort);
        }
        let head = &data[..8];
        let kind = if head == self.create {
            EventKind::Create
        } else if head == self.trade {
            EventKind::Trade
        } else if head == self.complete {
            EventKind::Complete
        } else {
            self.rejected += 1;
            return Err(Reject::UnknownDiscriminator);
        };
        self.passed += 1;
        Ok(kind)
    }

    pub fn reject_rate(&self) -> Option<f64> {
        let total = self.passed + self.rejected;
        if total == 0 {
            // Nothing observed is not a 0% reject rate.
            return None;
        }
        Some(self.rejected as f64 / total as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROGRAM: [u8; 32] = [7u8; 32];
    const CREATE: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    const TRADE: [u8; 8] = [8, 7, 6, 5, 4, 3, 2, 1];
    const COMPLETE: [u8; 8] = [9, 9, 9, 9, 9, 9, 9, 9];

    fn filter() -> L0Filter {
        L0Filter::new(PROGRAM, CREATE, TRADE, COMPLETE)
    }

    #[test]
    fn known_discriminators_are_classified() {
        let mut f = filter();
        let mut payload = TRADE.to_vec();
        payload.extend_from_slice(&[0u8; 64]);
        assert_eq!(f.classify(&PROGRAM, &payload), Ok(EventKind::Trade));
        assert_eq!(f.passed, 1);
    }

    #[test]
    fn every_other_program_is_dropped_first() {
        let mut f = filter();
        let mut payload = TRADE.to_vec();
        payload.extend_from_slice(&[0u8; 64]);
        assert_eq!(f.classify(&[1u8; 32], &payload), Err(Reject::ForeignProgram));
        assert_eq!(f.passed, 0);
    }

    #[test]
    fn a_truncated_payload_is_dropped() {
        let mut f = filter();
        assert_eq!(f.classify(&PROGRAM, &[1, 2, 3]), Err(Reject::TooShort));
    }

    #[test]
    fn an_unknown_discriminator_is_dropped() {
        let mut f = filter();
        let payload = [0xAAu8; 32];
        assert_eq!(f.classify(&PROGRAM, &payload), Err(Reject::UnknownDiscriminator));
    }

    #[test]
    fn nothing_observed_is_not_a_zero_reject_rate() {
        assert_eq!(filter().reject_rate(), None);
    }

    #[test]
    fn the_filter_passes_every_recognised_kind() {
        let mut f = filter();
        for (head, expected) in [
            (CREATE, EventKind::Create),
            (TRADE, EventKind::Trade),
            (COMPLETE, EventKind::Complete),
        ] {
            let mut payload = head.to_vec();
            payload.extend_from_slice(&[0u8; 32]);
            // It may pass what later stages reject; it must never reject what
            // they would have accepted.
            assert_eq!(f.classify(&PROGRAM, &payload), Ok(expected));
        }
        assert_eq!(f.passed, 3);
        assert_eq!(f.reject_rate(), Some(0.0));
    }
}
