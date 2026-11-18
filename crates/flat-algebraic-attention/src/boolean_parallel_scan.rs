//! BPS-12: bounded multi-threaded CPU execution of an exact Boolean prefix scan.
//!
//! Unlike classical hidden-state-gated LSTM, these input-only Boolean
//! transitions can be composed associatively. Each tree block at a given
//! prefix-scan level is independent; scoped threads operate on disjoint
//! mutable slices without locks or atomics.
//!
//! This is a research path, not a default backend or energy/speedup claim.

use std::thread;

use super::boolean_scan_lstm::{BooleanGateProgram, BooleanTransition};

/// Upper bound on simultaneously scheduled BPS-12 CPU workers.
pub const MAX_BOOLEAN_SCAN_WORKERS: usize = 8;

/// Exactly one Blelloch tree block. The two half-block tails are the
/// only elements modified at a given level.
fn update_block(block: &mut [BooleanTransition], stride: usize, up_sweep: bool) {
    let (left, right) = block.split_at_mut(stride);
    let left_end = &mut left[stride - 1];
    let right_end = &mut right[stride - 1];
    if up_sweep {
        *right_end = (*left_end).then(*right_end);
    } else {
        let parent_prefix = *right_end;
        let left_subtree = *left_end;
        *left_end = parent_prefix;
        *right_end = parent_prefix.then(left_subtree);
    }
}

/// One stage, either serial or using scoped workers over disjoint chunks.
/// Even with many CPUs, small stages remain serial to limit spawn overhead.
fn stage(tree: &mut [BooleanTransition], stride: usize, up_sweep: bool, workers: usize) {
    let step = stride * 2;
    let groups = tree.len() / step;
    if workers <= 1 || tree.len() < 2048 || groups < workers * 4 {
        for block in tree.chunks_exact_mut(step) {
            update_block(block, stride, up_sweep);
        }
        return;
    }

    // Each chunk starts and ends at an integral tree-block boundary.
    // There are at most 'workers' scoped tasks for this stage.
    let groups_per_worker = groups.div_ceil(workers);
    let chunk_len = groups_per_worker * step;
    thread::scope(|scope| {
        for chunk in tree.chunks_mut(chunk_len) {
            scope.spawn(move || {
                for block in chunk.chunks_exact_mut(step) {
                    update_block(block, stride, up_sweep);
                }
            });
        }
    });
}

impl BooleanGateProgram {
    /// A work-efficient parallel tree scan, preserving BPS-1 output bits.
    ///
    /// 'requested_workers' is clamped to 1..=8. Setting 0 requests the
    /// serial algorithm; it never causes unbounded thread creation.
    /// The algorithm performs O(T) compositions and needs O(T) temporary
    /// transitions. The theoretical stage dependency depth is O(log T).
    /// Worker creation and scheduling still carry real CPU costs.
    #[must_use]
    pub fn forward_scan_parallel(
        &self,
        inputs: &[u64],
        initial_state: u64,
        requested_workers: usize,
    ) -> Vec<u64> {
        let length = inputs.len();
        if length == 0 {
            return Vec::new();
        }
        let Some(capacity) = length.checked_next_power_of_two() else {
            return self.forward_serial(inputs, initial_state);
        };
        let workers = requested_workers.clamp(1, MAX_BOOLEAN_SCAN_WORKERS);
        let mut tree = vec![BooleanTransition::IDENTITY; capacity];
        for (&input, slot) in inputs.iter().zip(tree.iter_mut()) {
            *slot = self.transition(input);
        }

        let mut stride = 1usize;
        while stride < capacity {
            stage(&mut tree, stride, true, workers);
            stride *= 2;
        }

        tree[capacity - 1] = BooleanTransition::IDENTITY;
        stride = capacity / 2;
        while stride > 0 {
            stage(&mut tree, stride, false, workers);
            stride /= 2;
        }

        inputs
            .iter()
            .zip(tree.iter())
            .map(|(&input, &exclusive)| {
                exclusive.then(self.transition(input)).apply(initial_state)
                    & self.output.evaluate(input)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean_scan_lstm::UnaryGate;

    fn next_random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn gate(seed: &mut u64) -> UnaryGate {
        UnaryGate {
            when_false: next_random(seed),
            when_true: next_random(seed),
        }
    }

    #[test]
    fn parallel_tree_matches_serial_and_work_efficient_at_all_boundaries() {
        let mut seed = 0x8e52_71b0_19dd_5a6fu64;
        for length in [
            0usize, 1, 2, 3, 7, 16, 31, 64, 65, 127, 128, 129, 1025, 2048, 4095, 4096, 4097, 8191,
        ] {
            let inputs: Vec<u64> = (0..length).map(|_| next_random(&mut seed)).collect();
            for _ in 0..3 {
                let program = BooleanGateProgram {
                    forget: gate(&mut seed),
                    write: gate(&mut seed),
                    output: gate(&mut seed),
                };
                let initial = next_random(&mut seed);
                let expected = program.forward_serial(&inputs, initial);
                assert_eq!(
                    program.forward_scan_work_efficient(&inputs, initial),
                    expected,
                    "baseline length={length}"
                );
                for workers in [0usize, 1, 2, 3, 4, 8, 32] {
                    assert_eq!(
                        program.forward_scan_parallel(&inputs, initial, workers),
                        expected,
                        "length={length} workers={workers}"
                    );
                }
            }
        }
    }

    #[test]
    fn each_level_matches_true_serial_even_when_operations_do_not_commute() {
        // A later clear must dominate an earlier set, including when the
        // overwrite falls on a different tree partition.
        let program = BooleanGateProgram {
            forget: UnaryGate {
                when_false: u64::MAX,
                when_true: 0,
            },
            write: UnaryGate {
                when_false: u64::MAX,
                when_true: 0,
            },
            output: UnaryGate {
                when_false: u64::MAX,
                when_true: u64::MAX,
            },
        };
        let inputs: Vec<u64> = (0..4097)
            .map(|i| if i % 31 == 0 { u64::MAX } else { 0 })
            .collect();
        for initial in [0, u64::MAX, 1, 1u64 << 63] {
            let expected = program.forward_serial(&inputs, initial);
            for workers in [1, 2, 8, 64] {
                assert_eq!(
                    program.forward_scan_parallel(&inputs, initial, workers),
                    expected
                );
            }
        }
    }

    #[test]
    fn blocks_are_disjoint_and_parallel_eligibility_is_bounded() {
        assert_eq!(MAX_BOOLEAN_SCAN_WORKERS, 8);
        let mut tree: Vec<BooleanTransition> = (0..4096)
            .map(|i| BooleanTransition {
                keep: if i % 2 == 0 { u64::MAX } else { 0 },
                set: i as u64,
            })
            .collect();
        let mut serial = tree.clone();
        stage(&mut tree, 1, true, 8);
        stage(&mut serial, 1, true, 1);
        assert_eq!(tree, serial);
        stage(&mut tree, 1, false, 8);
        stage(&mut serial, 1, false, 1);
        assert_eq!(tree, serial);
    }
}
