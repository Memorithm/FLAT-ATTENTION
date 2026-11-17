//! Research-only bit-sliced Boolean gated memory (BPS-LSTM candidate).
//!
//! This is LSTM-inspired, not an exact binarization of classical LSTM:
//! gates depend on current input, not recurrent state, and Boolean OR
//! replaces additive memory. This restriction admits an associative scan.
//! Each bit of a u64 is an independent lane, not a 64-bit floating value.
//! Training uses deterministic Hamming-loss coordinate descent, not BPTT.
//! No latency, accuracy or energy advantage is claimed.

use std::thread;

/// A trainable unary gate, with one truth-table bit per lane and input value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnaryGate {
    pub when_false: u64,
    pub when_true: u64,
}

impl UnaryGate {
    #[must_use]
    pub const fn evaluate(self, input: u64) -> u64 {
        (!input & self.when_false) | (input & self.when_true)
    }
}

/// Boolean affine map on a packed memory word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BooleanTransition {
    pub keep: u64,
    pub set: u64,
}

impl BooleanTransition {
    pub const IDENTITY: Self = Self {
        keep: u64::MAX,
        set: 0,
    };

    #[must_use]
    pub const fn apply(self, state: u64) -> u64 {
        (self.keep & state) | self.set
    }

    /// Apply self first, then next; the operator is associative.
    #[must_use]
    pub const fn then(self, next: Self) -> Self {
        Self {
            keep: next.keep & self.keep,
            set: next.set | (next.keep & self.set),
        }
    }
}

/// Input-only gates: forgetting, fused writing, and exposing the memory.
///
/// The 64 lanes are independent. There is no hidden-state-dependent gating
/// or cross-lane synaptic coupling in this initial research candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BooleanGateProgram {
    pub forget: UnaryGate,
    pub write: UnaryGate,
    pub output: UnaryGate,
}

impl Default for BooleanGateProgram {
    fn default() -> Self {
        Self {
            forget: UnaryGate::default(),
            write: UnaryGate::default(),
            output: UnaryGate {
                when_false: u64::MAX,
                when_true: u64::MAX,
            },
        }
    }
}

impl BooleanGateProgram {
    #[must_use]
    pub const fn transition(self, input: u64) -> BooleanTransition {
        BooleanTransition {
            keep: self.forget.evaluate(input),
            set: self.write.evaluate(input),
        }
    }

    /// O(T) reference path with constant auxiliary memory.
    #[must_use]
    pub fn forward_serial(&self, inputs: &[u64], initial_state: u64) -> Vec<u64> {
        let mut state = initial_state;
        inputs
            .iter()
            .copied()
            .map(|input| {
                state = self.transition(input).apply(state);
                state & self.output.evaluate(input)
            })
            .collect()
    }

    /// Hillis-Steele scan: O(T log T) work, O(T) memory and O(log T)
    /// dependency depth. Each scan stage runs on scoped CPU threads
    /// when T >= 128 and more than one logical CPU is available.
    /// Extra scan work and scheduling can be slower than the serial path.
    #[must_use]
    pub fn forward_scan(&self, inputs: &[u64], initial_state: u64) -> Vec<u64> {
        let n = inputs.len();
        let mut prefixes: Vec<BooleanTransition> =
            inputs.iter().map(|&input| self.transition(input)).collect();

        let workers = thread::available_parallelism().map_or(1, |n| n.get().min(8));
        let mut stride = 1usize;
        while stride < n {
            let mut next = vec![BooleanTransition::IDENTITY; n];
            if n >= 128 && workers > 1 {
                let chunk_size = n.div_ceil(workers);
                thread::scope(|scope| {
                    for (chunk_index, chunk) in next.chunks_mut(chunk_size).enumerate() {
                        let previous = &prefixes;
                        scope.spawn(move || {
                            let start = chunk_index * chunk_size;
                            for (offset, output) in chunk.iter_mut().enumerate() {
                                let index = start + offset;
                                *output = if index < stride {
                                    previous[index]
                                } else {
                                    previous[index - stride].then(previous[index])
                                };
                            }
                        });
                    }
                });
            } else {
                for index in 0..n {
                    next[index] = if index < stride {
                        prefixes[index]
                    } else {
                        prefixes[index - stride].then(prefixes[index])
                    };
                }
            }
            prefixes = next;
            if stride > n / 2 {
                break;
            }
            stride *= 2;
        }

        inputs
            .iter()
            .zip(prefixes.iter())
            .map(|(&input, &prefix)| {
                prefix.apply(initial_state) & self.output.evaluate(input)
            })
            .collect()
    }

    /// Logical transitions only, NOT a power or joule measurement.
    #[must_use]
    pub fn logical_activity(
        &self,
        inputs: &[u64],
        initial_state: u64,
        active_lanes: u64,
    ) -> BooleanActivity {
        let mut activity = BooleanActivity::default();
        let mut state = initial_state;
        let mut previous_output = 0;
        for &input in inputs {
            let transition = self.transition(input);
            let next = transition.apply(state);
            let output = next & self.output.evaluate(input);
            activity.state_flips += u128::from(((next ^ state) & active_lanes).count_ones());
            activity.output_flips +=
                u128::from(((output ^ previous_output) & active_lanes).count_ones());
            activity.requested_write_bits +=
                u128::from((transition.set & active_lanes).count_ones());
            activity.active_output_bits += u128::from((output & active_lanes).count_ones());
            state = next;
            previous_output = output;
        }
        activity
    }

    /// Supervised Hamming loss, using only integer and Boolean operations.
    pub fn hamming_loss(
        &self,
        examples: &[BooleanTrainingSequence<'_>],
        active_lanes: u64,
    ) -> Result<u64, BooleanTrainError> {
        validate_examples(examples, active_lanes)?;
        self.loss_validated(examples, active_lanes)
    }

    /// Greedy Boolean parameter flips, accepting only strict loss reductions.
    /// This baseline learning rule gives no global convergence guarantee.
    pub fn fit_hamming(
        &mut self,
        examples: &[BooleanTrainingSequence<'_>],
        active_lanes: u64,
        max_epochs: usize,
    ) -> Result<BooleanFitReport, BooleanTrainError> {
        validate_examples(examples, active_lanes)?;
        if max_epochs == 0 {
            return Err(BooleanTrainError::ZeroEpochs);
        }

        let mut best = self.loss_validated(examples, active_lanes)?;
        let initial_loss = best;
        let mut accepted_flips = 0u64;
        let mut epochs = 0usize;
        for _ in 0..max_epochs {
            epochs += 1;
            let mut improved = false;
            for parameter in 0..6 {
                for lane in 0..64 {
                    let bit = 1u64 << lane;
                    if active_lanes & bit == 0 {
                        continue;
                    }
                    let mut candidate = *self;
                    candidate.toggle(parameter, bit);
                    let loss = candidate.loss_validated(examples, active_lanes)?;
                    if loss < best {
                        *self = candidate;
                        best = loss;
                        accepted_flips += 1;
                        improved = true;
                    }
                }
            }
            if !improved || best == 0 {
                break;
            }
        }
        Ok(BooleanFitReport {
            initial_loss,
            final_loss: best,
            accepted_flips,
            epochs,
        })
    }

    fn toggle(&mut self, parameter: usize, bit: u64) {
        match parameter {
            0 => self.forget.when_false ^= bit,
            1 => self.forget.when_true ^= bit,
            2 => self.write.when_false ^= bit,
            3 => self.write.when_true ^= bit,
            4 => self.output.when_false ^= bit,
            5 => self.output.when_true ^= bit,
            _ => unreachable!("only six Boolean parameters exist"),
        }
    }

    fn loss_validated(
        &self,
        examples: &[BooleanTrainingSequence<'_>],
        active_lanes: u64,
    ) -> Result<u64, BooleanTrainError> {
        let mut loss = 0u64;
        for example in examples {
            let mut state = example.initial_state;
            for (&input, &target) in example.inputs.iter().zip(example.targets.iter()) {
                state = self.transition(input).apply(state);
                let output = state & self.output.evaluate(input);
                let errors = u64::from(((output ^ target) & active_lanes).count_ones());
                loss = loss.checked_add(errors).ok_or(BooleanTrainError::LossOverflow)?;
            }
        }
        Ok(loss)
    }
}

/// Operation-level proxy, deliberately not translated into energy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BooleanActivity {
    pub state_flips: u128,
    pub output_flips: u128,
    pub requested_write_bits: u128,
    pub active_output_bits: u128,
}

#[derive(Clone, Copy, Debug)]
pub struct BooleanTrainingSequence<'a> {
    pub inputs: &'a [u64],
    pub targets: &'a [u64],
    pub initial_state: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanTrainError {
    EmptyDataset,
    EmptyActiveLanes,
    EmptySequence { example_index: usize },
    LengthMismatch { example_index: usize },
    ZeroEpochs,
    LossOverflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BooleanFitReport {
    pub initial_loss: u64,
    pub final_loss: u64,
    pub accepted_flips: u64,
    pub epochs: usize,
}

fn validate_examples(
    examples: &[BooleanTrainingSequence<'_>],
    active_lanes: u64,
) -> Result<(), BooleanTrainError> {
    if examples.is_empty() {
        return Err(BooleanTrainError::EmptyDataset);
    }
    if active_lanes == 0 {
        return Err(BooleanTrainError::EmptyActiveLanes);
    }
    for (index, example) in examples.iter().enumerate() {
        if example.inputs.is_empty() {
            return Err(BooleanTrainError::EmptySequence {
                example_index: index,
            });
        }
        if example.inputs.len() != example.targets.len() {
            return Err(BooleanTrainError::LengthMismatch {
                example_index: index,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn transition_composition_is_associative_and_ordered() {
        let a = BooleanTransition { keep: 0, set: 1 };
        let b = BooleanTransition { keep: 0, set: 0 };
        assert_eq!(a.then(b).apply(0), 0);
        assert_eq!(b.then(a).apply(0), 1);

        let mut seed = 0x1357_9bdf_2468_ace1u64;
        for _ in 0..512 {
            let a = BooleanTransition {
                keep: next_random(&mut seed),
                set: next_random(&mut seed),
            };
            let b = BooleanTransition {
                keep: next_random(&mut seed),
                set: next_random(&mut seed),
            };
            let c = BooleanTransition {
                keep: next_random(&mut seed),
                set: next_random(&mut seed),
            };
            let state = next_random(&mut seed);
            assert_eq!(a.then(b).then(c), a.then(b.then(c)));
            assert_eq!(a.then(b).then(c).apply(state), c.apply(b.apply(a.apply(state))));
            assert_eq!(a.then(BooleanTransition::IDENTITY), a);
            assert_eq!(BooleanTransition::IDENTITY.then(a), a);
        }
    }

    #[test]
    fn scan_matches_serial_for_non_power_of_two_lengths_and_initial_states() {
        let mut seed = 0xa5a5_33cc_7890_1234u64;
        for length in [0usize, 1, 2, 3, 5, 31, 64, 65, 127, 128, 257] {
            let inputs: Vec<u64> = (0..length).map(|_| next_random(&mut seed)).collect();
            for _ in 0..4 {
                let program = BooleanGateProgram {
                    forget: UnaryGate {
                        when_false: next_random(&mut seed),
                        when_true: next_random(&mut seed),
                    },
                    write: UnaryGate {
                        when_false: next_random(&mut seed),
                        when_true: next_random(&mut seed),
                    },
                    output: UnaryGate {
                        when_false: next_random(&mut seed),
                        when_true: next_random(&mut seed),
                    },
                };
                let initial_state = next_random(&mut seed);
                assert_eq!(
                    program.forward_scan(&inputs, initial_state),
                    program.forward_serial(&inputs, initial_state),
                    "length={length}"
                );
            }
        }
    }

    #[test]
    fn activity_counts_logical_transitions() {
        let program = BooleanGateProgram {
            forget: UnaryGate::default(),
            write: UnaryGate {
                when_false: 0,
                when_true: 1,
            },
            output: UnaryGate {
                when_false: u64::MAX,
                when_true: u64::MAX,
            },
        };
        let activity = program.logical_activity(&[1, 1, 0], 0, 1);
        assert_eq!(activity.state_flips, 2);
        assert_eq!(activity.output_flips, 2);
        assert_eq!(activity.requested_write_bits, 2);
        assert_eq!(activity.active_output_bits, 2);
    }

    #[test]
    fn truth_table_covers_both_extreme_lanes() {
        let gate = UnaryGate {
            when_false: 1,
            when_true: 1u64 << 63,
        };
        assert_eq!(gate.evaluate(0), 1);
        assert_eq!(gate.evaluate(u64::MAX), 1u64 << 63);
    }

    #[test]
    fn discrete_training_learns_a_one_lane_input_copy() {
        let inputs = [0u64, 1, 0, 1, 1, 0];
        let example = BooleanTrainingSequence {
            inputs: &inputs,
            targets: &inputs,
            initial_state: 0,
        };
        let mut program = BooleanGateProgram::default();
        let report = program.fit_hamming(&[example], 1, 4).unwrap();
        assert!(report.initial_loss > 0);
        assert_eq!(report.final_loss, 0);
        assert_eq!(program.forward_serial(&inputs, 0), inputs);
        assert_eq!(program.forward_scan(&inputs, 0), inputs);
    }

    #[test]
    fn malformed_training_inputs_are_rejected() {
        let mut program = BooleanGateProgram::default();
        assert_eq!(
            program.fit_hamming(&[], 1, 1),
            Err(BooleanTrainError::EmptyDataset)
        );
        let empty = BooleanTrainingSequence {
            inputs: &[],
            targets: &[],
            initial_state: 0,
        };
        assert_eq!(
            program.fit_hamming(&[empty], 1, 1),
            Err(BooleanTrainError::EmptySequence { example_index: 0 })
        );
        let mismatch = BooleanTrainingSequence {
            inputs: &[0, 1],
            targets: &[0],
            initial_state: 0,
        };
        assert_eq!(
            program.hamming_loss(&[mismatch], 1),
            Err(BooleanTrainError::LengthMismatch { example_index: 0 })
        );
        let example = BooleanTrainingSequence {
            inputs: &[1],
            targets: &[1],
            initial_state: 0,
        };
        assert_eq!(
            program.fit_hamming(&[example], 0, 1),
            Err(BooleanTrainError::EmptyActiveLanes)
        );
        assert_eq!(
            program.fit_hamming(&[example], 1, 0),
            Err(BooleanTrainError::ZeroEpochs)
        );
    }
}
