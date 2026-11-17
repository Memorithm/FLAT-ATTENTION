//! Lane-parallel discrete learning for the research-only Boolean scan memory.
//!
//! Errors and Hamming counters are kept in bit planes. One candidate evaluation
//! tests an entire 64-lane Boolean parameter mask at once; no floating point.

use super::boolean_scan_lstm::{
    validate_examples, BooleanFitReport, BooleanGateProgram, BooleanTrainError,
    BooleanTrainingSequence,
};

/// Binary ripple-counter planes. Plane k stores count bit k for all 64 lanes.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PackedLaneCounts {
    planes: [u64; 64],
}

impl PackedLaneCounts {
    fn zero() -> Self {
        Self { planes: [0; 64] }
    }

    fn add_mask(&mut self, mut carry: u64) -> Result<(), BooleanTrainError> {
        for plane in &mut self.planes {
            if carry == 0 {
                return Ok(());
            }
            let next_carry = *plane & carry;
            *plane ^= carry;
            carry = next_carry;
        }
        if carry != 0 {
            return Err(BooleanTrainError::LossOverflow);
        }
        Ok(())
    }

    /// For each lane, select 1 iff candidate error count is strictly smaller.
    fn better_lanes(&self, candidate: &Self) -> u64 {
        let mut equal = u64::MAX;
        let mut better = 0u64;
        for index in (0..64).rev() {
            let current_bit = self.planes[index];
            let candidate_bit = candidate.planes[index];
            better |= equal & current_bit & !candidate_bit;
            equal &= !(current_bit ^ candidate_bit);
        }
        better
    }

    fn replace_better_lanes(&mut self, candidate: &Self, better: u64) {
        for (old, &new) in self.planes.iter_mut().zip(candidate.planes.iter()) {
            *old = (*old & !better) | (new & better);
        }
    }

    fn total(&self, active_lanes: u64) -> Result<u64, BooleanTrainError> {
        let mut total = 0u128;
        for (index, &plane) in self.planes.iter().enumerate() {
            total += u128::from((plane & active_lanes).count_ones()) << index;
        }
        u64::try_from(total).map_err(|_| BooleanTrainError::LossOverflow)
    }
}

fn loss_planes(
    program: &BooleanGateProgram,
    examples: &[BooleanTrainingSequence<'_>],
    active_lanes: u64,
) -> Result<PackedLaneCounts, BooleanTrainError> {
    let mut losses = PackedLaneCounts::zero();
    for example in examples {
        let mut state = example.initial_state;
        for (&input, &target) in example.inputs.iter().zip(example.targets.iter()) {
            state = program.transition(input).apply(state);
            let output = state & program.output.evaluate(input);
            losses.add_mask((output ^ target) & active_lanes)?;
        }
    }
    Ok(losses)
}

fn flip_parameter(program: &mut BooleanGateProgram, parameter: usize, mask: u64) {
    match parameter {
        0 => program.forget.when_false ^= mask,
        1 => program.forget.when_true ^= mask,
        2 => program.write.when_false ^= mask,
        3 => program.write.when_true ^= mask,
        4 => program.output.when_false ^= mask,
        5 => program.output.when_true ^= mask,
        _ => unreachable!("only six Boolean parameter masks exist"),
    }
}

impl BooleanGateProgram {
    /// Train 64 independent lanes simultaneously with bit-sliced Hamming
    /// counters and Boolean comparisons, with no floating point or gradients.
    ///
    /// Compared with per-lane greedy training, each parameter direction needs
    /// only one candidate forward evaluation, rather than up to 64. This is
    /// algorithmic savings, not a benchmark or an energy measurement.
    ///
    /// This equivalence relies strictly on absent cross-lane connections.
    pub fn fit_hamming_parallel_lanes(
        &mut self,
        examples: &[BooleanTrainingSequence<'_>],
        active_lanes: u64,
        max_epochs: usize,
    ) -> Result<BooleanFitReport, BooleanTrainError> {
        validate_examples(examples, active_lanes)?;
        if max_epochs == 0 {
            return Err(BooleanTrainError::ZeroEpochs);
        }

        let mut counts = loss_planes(self, examples, active_lanes)?;
        let initial_loss = counts.total(active_lanes)?;
        let mut final_loss = initial_loss;
        let mut accepted_flips = 0u64;
        let mut epochs = 0usize;

        for _ in 0..max_epochs {
            epochs += 1;
            let mut improved = false;
            for parameter in 0..6 {
                let mut candidate = *self;
                flip_parameter(&mut candidate, parameter, active_lanes);
                let candidate_counts = loss_planes(&candidate, examples, active_lanes)?;
                let better = counts.better_lanes(&candidate_counts) & active_lanes;
                if better != 0 {
                    flip_parameter(self, parameter, better);
                    counts.replace_better_lanes(&candidate_counts, better);
                    accepted_flips += u64::from(better.count_ones());
                    improved = true;
                }
            }
            final_loss = counts.total(active_lanes)?;
            if !improved || final_loss == 0 {
                break;
            }
        }

        Ok(BooleanFitReport {
            initial_loss,
            final_loss,
            accepted_flips,
            epochs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_counters_match_simple_hamming_counts() {
        let mut counters = PackedLaneCounts::zero();
        let mask = (1u64 << 63) | 1;
        for _ in 0..257 {
            counters.add_mask(mask).unwrap();
        }
        assert_eq!(counters.total(mask), 514);
        assert_eq!(counters.total(1), 257);

        let mut lower = PackedLaneCounts::zero();
        for _ in 0..256 {
            lower.add_mask(mask).unwrap();
        }
        assert_eq!(counters.better_lanes(&lower) & mask, mask);
        assert_eq!(lower.better_lanes(&counters) & mask, 0);
    }

    #[test]
    fn parallel_lanes_training_matches_coordinate_descent_on_all_lanes() {
        let inputs = [
            0x0123_4567_89ab_cdefu64,
            0xfedc_ba98_7654_3210,
            0,
            u64::MAX,
            0x3333_3333_3333_3333,
            0x5555_5555_5555_5555,
        ];
        let dataset = [BooleanTrainingSequence {
            inputs: &inputs,
            targets: &inputs,
            initial_state: 0,
        }];
        let mut serial_trainer = BooleanGateProgram::default();
        let mut parallel_trainer = serial_trainer;

        let serial_report = serial_trainer.fit_hamming(&dataset, u64::MAX, 3).unwrap();
        let parallel_report = parallel_trainer
            .fit_hamming_parallel_lanes(&dataset, u64::MAX, 3)
            .unwrap();
        assert_eq!(serial_trainer, parallel_trainer);
        assert_eq!(serial_report, parallel_report);
        assert_eq!(parallel_report.final_loss, 0);
        assert_eq!(parallel_trainer.forward_scan(&inputs, 0), inputs);
    }

    #[test]
    fn inactive_lanes_and_errors_remain_unchanged() {
        let inputs = [0u64, 1, 0, 1];
        let dataset = [BooleanTrainingSequence {
            inputs: &inputs,
            targets: &inputs,
            initial_state: 0,
        }];
        let mut program = BooleanGateProgram::default();
        assert_eq!(
            program.fit_hamming_parallel_lanes(&dataset, 0, 2),
            Err(BooleanTrainError::EmptyActiveLanes)
        );
        let report = program
            .fit_hamming_parallel_lanes(&dataset, 1, 2)
            .unwrap();
        assert_eq!(report.final_loss, 0);
        assert_eq!(program.write.when_true & !1u64, 0);
    }
}
