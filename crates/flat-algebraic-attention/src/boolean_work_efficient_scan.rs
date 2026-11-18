//! BPS-11: work-efficient exact prefix scan for independent Boolean LSTM lanes.
//!
//! This is a second execution strategy for the existing input-gated, packed
//! Boolean memory. It does NOT change its truth tables or recurrent semantics.
//! Each up/down tree level is independent and may be parallelized in a later
//! measured implementation; this CPU reference intentionally stays serial.
//! No wall-clock, accuracy, or energy advantage is implied.

use super::boolean_scan_lstm::{BooleanGateProgram, BooleanTransition};

impl BooleanGateProgram {
    /// Blelloch exclusive scan followed by a single inclusive transition.
    ///
    /// Uses O(T) Boolean-transition compositions and O(T) auxiliary memory,
    /// with O(log T) *algorithmic* dependency depth. Padding is by the exact
    /// identity transition and therefore preserves non-power-of-two lengths.
    /// Unlike `forward_scan` (Hillis-Steele), this routine does not spawn
    /// threads: work efficiency alone does not constitute a CPU speedup.
    #[must_use]
    pub fn forward_scan_work_efficient(&self, inputs: &[u64], initial_state: u64) -> Vec<u64> {
        let n = inputs.len();
        if n == 0 {
            return Vec::new();
        }
        let Some(capacity) = n.checked_next_power_of_two() else {
            // An unrepresentable padded capacity must not panic or overflow.
            return self.forward_serial(inputs, initial_state);
        };
        let mut prefixes = vec![BooleanTransition::IDENTITY; capacity];
        for (slot, &input) in prefixes.iter_mut().zip(inputs) {
            *slot = self.transition(input);
        }

        // Up-sweep: each parent stores the ordered composition of its left
        // subtree followed by its right subtree. 'then' is associative but
        // NOT commutative; the order is semantically significant.
        let mut stride = 1usize;
        while stride < capacity {
            let step = stride * 2;
            for end in (step - 1..capacity).step_by(step) {
                let left = end - stride;
                prefixes[end] = prefixes[left].then(prefixes[end]);
            }
            stride = step;
        }

        // Down-sweep: replace the root with identity and distribute EXCLUSIVE
        // prefixes. The right child must include its left subtree after the
        // prefix inherited from the parent.
        prefixes[capacity - 1] = BooleanTransition::IDENTITY;
        let mut stride = capacity / 2;
        while stride > 0 {
            let step = stride * 2;
            for end in (step - 1..capacity).step_by(step) {
                let left = end - stride;
                let parent_prefix = prefixes[end];
                let left_total = prefixes[left];
                prefixes[left] = parent_prefix;
                prefixes[end] = parent_prefix.then(left_total);
            }
            stride /= 2;
        }

        inputs
            .iter()
            .copied()
            .enumerate()
            .map(|(index, input)| {
                let inclusive = prefixes[index].then(self.transition(input));
                inclusive.apply(initial_state) & self.output.evaluate(input)
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

    fn random_gate(seed: &mut u64) -> UnaryGate {
        UnaryGate {
            when_false: next_random(seed),
            when_true: next_random(seed),
        }
    }

    #[test]
    fn work_efficient_matches_serial_for_irregular_lengths_and_64_lanes() {
        let mut seed = 0x27a8_14bb_98d3_6ec1u64;
        for length in [
            0usize, 1, 2, 3, 5, 7, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 257, 513, 1025,
        ] {
            let inputs: Vec<u64> = (0..length).map(|_| next_random(&mut seed)).collect();
            for _ in 0..8 {
                let program = BooleanGateProgram {
                    forget: random_gate(&mut seed),
                    write: random_gate(&mut seed),
                    output: random_gate(&mut seed),
                };
                let initial = next_random(&mut seed);
                let serial = program.forward_serial(&inputs, initial);
                let scanned = program.forward_scan_work_efficient(&inputs, initial);
                assert_eq!(scanned, serial, "length={length}");
            }
        }
    }

    #[test]
    fn agrees_with_existing_hillis_steele_reference() {
        let mut seed = 0x8756_4fea_27bc_1039u64;
        let program = BooleanGateProgram {
            forget: random_gate(&mut seed),
            write: random_gate(&mut seed),
            output: random_gate(&mut seed),
        };
        for length in [0usize, 1, 3, 31, 64, 65, 129, 257] {
            let inputs: Vec<u64> = (0..length).map(|_| next_random(&mut seed)).collect();
            let initial = next_random(&mut seed);
            assert_eq!(
                program.forward_scan_work_efficient(&inputs, initial),
                program.forward_scan(&inputs, initial),
                "length={length}"
            );
        }
    }

    #[test]
    fn overwrite_and_retention_transitions_preserve_order_and_initial_state() {
        let program = BooleanGateProgram {
            forget: UnaryGate {
                when_false: u64::MAX,
                when_true: 0,
            },
            write: UnaryGate {
                when_false: 0,
                when_true: u64::MAX,
            },
            output: UnaryGate {
                when_false: u64::MAX,
                when_true: u64::MAX,
            },
        };
        let inputs = [0, u64::MAX, 0, 0, u64::MAX, 0, 0];
        for initial in [0, u64::MAX, 1, 1u64 << 63] {
            assert_eq!(
                program.forward_scan_work_efficient(&inputs, initial),
                program.forward_serial(&inputs, initial)
            );
        }
    }
}
