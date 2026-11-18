//! BPS-13: bounded, work-efficient prefix scan for cross-lane Boolean recurrence.
//!
//! BPS-12 parallelized only independent one-bit lanes. This path applies the
//! same work-efficient scan to BPS-2/BPS-3's genuinely connected state lanes.
//! A rotating transition is an associative, order-sensitive map over 64 bits.
//! No neural-model or event semantics change; no latency or energy claim.

use std::thread;

use super::boolean_edge_memory::BooleanEdgeProgram;
use super::boolean_parallel_scan::MAX_BOOLEAN_SCAN_WORKERS;
use super::boolean_rotating_memory::{BooleanRotatingProgram, RotatingTransition};

/// Update the two ends of an independent Blelloch tree block.
fn update_rotating_block(block: &mut [RotatingTransition], stride: usize, up: bool) {
    let (left, right) = block.split_at_mut(stride);
    let left_end = &mut left[stride - 1];
    let right_end = &mut right[stride - 1];
    if up {
        *right_end = (*left_end).then(*right_end);
    } else {
        let parent_prefix = *right_end;
        let left_total = *left_end;
        *left_end = parent_prefix;
        *right_end = parent_prefix.then(left_total);
    }
}

fn rotating_stage(tree: &mut [RotatingTransition], stride: usize, up: bool, workers: usize) {
    let step = stride * 2;
    let groups = tree.len() / step;
    if workers <= 1 || tree.len() < 2048 || groups < workers * 4 {
        for block in tree.chunks_exact_mut(step) {
            update_rotating_block(block, stride, up);
        }
        return;
    }

    let groups_per_worker = groups.div_ceil(workers);
    let chunk_len = groups_per_worker * step;
    thread::scope(|scope| {
        for chunk in tree.chunks_mut(chunk_len) {
            scope.spawn(move || {
                for block in chunk.chunks_exact_mut(step) {
                    update_rotating_block(block, stride, up);
                }
            });
        }
    });
}

/// Exact prefix scan for ordered rotating Boolean transition maps.
///
/// A map is `(keep & state.rotate_left(shift)) | set`. Composition retains
/// that form, so an identity-padded Blelloch scan uses O(E) transition work,
/// O(E) memory and O(log E) dependency depth for E maps. Up to eight bounded
/// CPU workers process nonoverlapping blocks at eligible tree levels.
///
/// Returns the state AFTER each transition. Empty input returns no states.
/// For an unrepresentable padded length, uses a serial exact fallback.
/// This performs no event detection, sampling or secret-key operations.
#[must_use]
pub fn scan_rotating_states_parallel(
    transitions: &[RotatingTransition],
    initial_state: u64,
    requested_workers: usize,
) -> Vec<u64> {
    let length = transitions.len();
    if length == 0 {
        return Vec::new();
    }
    let Some(capacity) = length.checked_next_power_of_two() else {
        let mut current = initial_state;
        return transitions
            .iter()
            .map(|transition| {
                current = transition.apply(current);
                current
            })
            .collect();
    };
    let workers = requested_workers.clamp(1, MAX_BOOLEAN_SCAN_WORKERS);
    let mut tree = vec![RotatingTransition::IDENTITY; capacity];
    tree[..length].copy_from_slice(transitions);

    let mut stride = 1usize;
    while stride < capacity {
        rotating_stage(&mut tree, stride, true, workers);
        stride *= 2;
    }

    tree[capacity - 1] = RotatingTransition::IDENTITY;
    stride = capacity / 2;
    while stride > 0 {
        rotating_stage(&mut tree, stride, false, workers);
        stride /= 2;
    }

    transitions
        .iter()
        .zip(tree.iter())
        .map(|(&transition, &exclusive)| exclusive.then(transition).apply(initial_state))
        .collect()
}

impl BooleanRotatingProgram {
    /// BPS-2: work-efficient CPU scan with actual cross-lane recurrent shifts.
    #[must_use]
    pub fn forward_scan_parallel(
        &self,
        inputs: &[u64],
        initial_state: u64,
        requested_workers: usize,
    ) -> Vec<u64> {
        let transitions: Vec<_> = inputs.iter().map(|&input| self.transition(input)).collect();
        scan_rotating_states_parallel(&transitions, initial_state, requested_workers)
            .into_iter()
            .zip(inputs.iter().copied())
            .map(|(state, input)| state & self.gates.output.evaluate(input))
            .collect()
    }
}

impl BooleanEdgeProgram {
    /// BPS-3 dense sampled reference using the work-efficient rotating scan.
    ///
    /// Every input tick is inspected and contributes an identity transition
    /// when there is no rising edge. This path does NOT skip quiet scan work.
    #[must_use]
    pub fn forward_scan_parallel(
        self,
        inputs: &[u64],
        baseline_input: u64,
        initial_state: u64,
        requested_workers: usize,
    ) -> Vec<u64> {
        let mut previous = baseline_input;
        let transitions: Vec<_> = inputs
            .iter()
            .map(|&current| {
                let rising = current & !previous;
                previous = current;
                self.transition(rising)
            })
            .collect();
        scan_rotating_states_parallel(&transitions, initial_state, requested_workers)
            .into_iter()
            .map(|state| state & self.readout_mask)
            .collect()
    }

    /// BPS-3 event-sparse path: scan only rising-front transitions.
    ///
    /// Event detection and the requested dense output remain O(T) for T
    /// sampled ticks, while scan composition uses O(E) for E eventful ticks
    /// rather than the prior O(E log E) Hillis-Steele implementation.
    /// Quiet ticks do not cause a rotation or gate evaluation, including
    /// across event boundaries and after a nonzero baseline sample.
    #[must_use]
    pub fn forward_event_scan_parallel(
        self,
        inputs: &[u64],
        baseline_input: u64,
        initial_state: u64,
        requested_workers: usize,
    ) -> Vec<u64> {
        let mut previous = baseline_input;
        let mut event_positions = Vec::new();
        let mut transitions = Vec::new();
        for (index, &current) in inputs.iter().enumerate() {
            let rising = current & !previous;
            previous = current;
            if rising != 0 {
                event_positions.push(index);
                transitions.push(self.transition(rising));
            }
        }

        let event_states =
            scan_rotating_states_parallel(&transitions, initial_state, requested_workers);
        let mut outputs = vec![initial_state & self.readout_mask; inputs.len()];
        for (index, &start) in event_positions.iter().enumerate() {
            let end = event_positions
                .get(index + 1)
                .copied()
                .unwrap_or(inputs.len());
            outputs[start..end].fill(event_states[index] & self.readout_mask);
        }
        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean_rotating_memory::scan_rotating_states;
    use crate::boolean_scan_lstm::{BooleanGateProgram, UnaryGate};

    fn random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn gate(seed: &mut u64) -> UnaryGate {
        UnaryGate {
            when_false: random(seed),
            when_true: random(seed),
        }
    }

    #[test]
    fn rotating_composition_matches_serial_scan_with_mixed_shifts() {
        let mut seed = 0x9d21_56b0_741c_0ae3u64;
        for length in [0usize, 1, 2, 3, 7, 31, 64, 65, 127, 129, 2048, 4097] {
            let transitions: Vec<_> = (0..length)
                .map(|_| {
                    RotatingTransition::new(
                        random(&mut seed),
                        random(&mut seed),
                        (random(&mut seed) % 64) as u8,
                    )
                    .unwrap()
                })
                .collect();
            let initial = random(&mut seed);
            let reference = scan_rotating_states(&transitions, initial);
            for workers in [0usize, 1, 2, 4, 8, 64] {
                assert_eq!(
                    scan_rotating_states_parallel(&transitions, initial, workers),
                    reference,
                    "mixed shifts length={length} workers={workers}"
                );
            }
        }
    }

    #[test]
    fn coupled_bps2_matches_serial_for_random_gates_and_rotations() {
        let mut seed = 0xa2ce_7519_338b_4dd1u64;
        for shift in [0, 1, 7, 31, 63] {
            for length in [0usize, 1, 3, 65, 129, 4097] {
                let program = BooleanRotatingProgram::new(
                    BooleanGateProgram {
                        forget: gate(&mut seed),
                        write: gate(&mut seed),
                        output: gate(&mut seed),
                    },
                    shift,
                )
                .unwrap();
                let inputs: Vec<u64> = (0..length).map(|_| random(&mut seed)).collect();
                let initial = random(&mut seed);
                let reference = program.forward_serial(&inputs, initial);
                for workers in [1, 2, 8, 64] {
                    assert_eq!(
                        program.forward_scan_parallel(&inputs, initial, workers),
                        reference,
                        "BPS-2 shift={shift} length={length} workers={workers}"
                    );
                }
            }
        }
    }

    #[test]
    fn rising_edge_event_sparse_matches_dense_and_old_scan() {
        let mut seed = 0x528e_911a_4c7b_d2f6u64;
        for shift in [0, 1, 7, 31, 63] {
            let program =
                BooleanEdgeProgram::new(gate(&mut seed), gate(&mut seed), random(&mut seed), shift)
                    .unwrap();
            for length in [0usize, 1, 3, 64, 65, 129, 2049, 4097] {
                let baseline = random(&mut seed);
                let initial = random(&mut seed);
                let mut held = baseline;
                let inputs: Vec<u64> = (0..length)
                    .map(|tick| {
                        if tick % 11 == 0 {
                            held = random(&mut seed);
                        } else if tick % 19 == 0 {
                            held = 0;
                        }
                        held
                    })
                    .collect();
                let reference = program.forward_dense(&inputs, baseline, initial);
                assert_eq!(
                    program.forward_event_scan(&inputs, baseline, initial),
                    reference
                );
                for workers in [0usize, 1, 2, 8, 64] {
                    assert_eq!(
                        program.forward_scan_parallel(&inputs, baseline, initial, workers),
                        reference,
                        "BPS-3 dense shift={shift} length={length} workers={workers}"
                    );
                    assert_eq!(
                        program.forward_event_scan_parallel(&inputs, baseline, initial, workers),
                        reference,
                        "BPS-3 sparse shift={shift} length={length} workers={workers}"
                    );
                }
            }
        }
    }

    #[test]
    fn quiet_ticks_and_rearming_do_not_rotate_state() {
        let always = UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX,
        };
        let program = BooleanEdgeProgram::new(always, UnaryGate::default(), u64::MAX, 1).unwrap();
        let inputs = [1, 1, 1, 0, 0, 1, 1, 0, 1];
        let expected = program.forward_dense(&inputs, 0, 1);
        assert_eq!(expected, vec![2, 2, 2, 2, 2, 4, 4, 4, 8]);
        assert_eq!(
            program.forward_event_scan_parallel(&inputs, 0, 1, 8),
            expected
        );
        assert_eq!(program.forward_scan_parallel(&inputs, 0, 1, 8), expected);
        assert_eq!(
            program.forward_event_scan_parallel(&[1, 1, 0, 1], 1, 1, 8),
            vec![1, 1, 1, 2]
        );
    }

    #[test]
    fn parallel_rotating_stage_matches_serial_for_noncommutative_maps() {
        let mut tree = (0..4096)
            .map(|index| {
                RotatingTransition::new(
                    if index % 5 == 0 { 0 } else { u64::MAX },
                    (index as u64).rotate_left(7),
                    (index % 64) as u8,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut serial = tree.clone();
        for stride in [1, 2, 4, 8, 16, 32] {
            rotating_stage(&mut tree, stride, true, 8);
            rotating_stage(&mut serial, stride, true, 1);
            assert_eq!(tree, serial, "up stride={stride}");
        }
        for stride in [32, 16, 8, 4, 2, 1] {
            rotating_stage(&mut tree, stride, false, 8);
            rotating_stage(&mut serial, stride, false, 1);
            assert_eq!(tree, serial, "down stride={stride}");
        }
    }
}
