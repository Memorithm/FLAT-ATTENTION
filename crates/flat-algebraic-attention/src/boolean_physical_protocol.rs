//! BPS-6: exact-front-density CPU trial protocol for trustworthy external measurement.
//!
//! A benchmark is not an energy measurement. Linux RAPL readers must be hosted
//! by a separate, trusted control plane such as Memorithm/RemoteOps; this Rust
//! module never reads /sys, perf events, wall power or privilege-sensitive data.
//! The output-copy negative control deliberately performs less semantic work.

use std::hint::black_box;
use std::time::Instant;

use super::boolean_edge_memory::BooleanEdgeProgram;
use super::boolean_scan_lstm::UnaryGate;

/// Fixed workload densities are fractions of *eventful sampled steps*,
/// not fractions of high input words that might remain high.
pub const TRIAL_FRONT_DENSITIES: [u8; 5] = [0, 1, 10, 50, 100];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrialError {
    InvalidSteps,
    InvalidDensity,
    InvalidRepetitions,
    OracleMismatch,
    WrongEventCount,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrialMode {
    OutputCopy,
    Dense,
    Sparse,
    EventScan,
}

impl TrialMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OutputCopy => "output_copy",
            Self::Dense => "dense",
            Self::Sparse => "sparse",
            Self::EventScan => "event_scan",
        }
    }

    #[must_use]
    pub const fn short_label(self) -> &'static str {
        match self {
            Self::OutputCopy => "COPY",
            Self::Dense => "DENSE",
            Self::Sparse => "SPARSE",
            Self::EventScan => "ESCAN",
        }
    }
}

/// The initial state and gates are frozen before any timing measurements.
#[must_use]
pub const fn trial_program() -> BooleanEdgeProgram {
    match BooleanEdgeProgram::new(
        UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX,
        },
        UnaryGate {
            when_false: 0,
            when_true: u64::MAX,
        },
        u64::MAX,
        7,
    ) {
        Ok(program) => program,
        Err(_) => panic!("constant BPS-6 rotation must be valid"),
    }
}

/// Deterministic one-hot sampled words with an exact number of rising events.
///
/// Adjacent eventful ticks use *different* bit positions, so each produces a
/// new 0->1 front even at 100% event-step density. Quiet ticks have a zero
/// input word. Baseline is zero. Every eventful tick emits exactly one front.
/// This generator is synthetic and its natural-language quality is untested.
pub fn exact_front_inputs(steps: usize, density_percent: u8) -> Result<Vec<u64>, TrialError> {
    if !(1..=65_536).contains(&steps) {
        return Err(TrialError::InvalidSteps);
    }
    if density_percent > 100 {
        return Err(TrialError::InvalidDensity);
    }
    let mut accumulator = 0u32;
    let mut inputs = Vec::with_capacity(steps);
    for tick in 0..steps {
        accumulator += u32::from(density_percent);
        if accumulator >= 100 {
            accumulator -= 100;
            inputs.push(1u64 << (tick % 64));
        } else {
            inputs.push(0);
        }
    }
    Ok(inputs)
}

/// Independent literal front definition, no cached detector state.
#[must_use]
pub fn count_rising_steps(inputs: &[u64], baseline: u64) -> u64 {
    let mut previous = baseline;
    let mut count = 0u64;
    for &input in inputs {
        if input & !previous != 0 {
            count += 1;
        }
        previous = input;
    }
    count
}

/// FNV-like fingerprint prevents confusion between workloads with identical
/// lengths and event counts. It is NOT a cryptographic provenance hash.
#[must_use]
pub fn workload_fingerprint(inputs: &[u64]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for (tick, &word) in inputs.iter().enumerate() {
        hash ^= word ^ (tick as u64).rotate_left(17);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedWorkload {
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
    pub event_steps: u64,
    pub fingerprint: u64,
}

/// Verify all four evaluation algorithms before allowing timed experiments.
/// The reference output is never computed inside the trial loop.
pub fn validated_workload(
    steps: usize,
    density_percent: u8,
) -> Result<ValidatedWorkload, TrialError> {
    let inputs = exact_front_inputs(steps, density_percent)?;
    let event_steps = count_rising_steps(&inputs, 0);
    let expected_events = (steps as u64 * u64::from(density_percent)) / 100;
    if event_steps != expected_events {
        return Err(TrialError::WrongEventCount);
    }
    let program = trial_program();
    let initial_state = 0x8000_0000_0000_0043u64;
    let outputs = program.forward_dense(&inputs, 0, initial_state);
    let sparse = program
        .forward_sparse(&inputs, 0, initial_state)
        .map_err(|_| TrialError::OracleMismatch)?;
    if sparse.activity.eventful_steps != u128::from(event_steps)
        || sparse.outputs != outputs
        || program.forward_event_scan(&inputs, 0, initial_state) != outputs
        || program.forward_scan(&inputs, 0, initial_state) != outputs
    {
        return Err(TrialError::OracleMismatch);
    }
    Ok(ValidatedWorkload {
        fingerprint: workload_fingerprint(&inputs),
        inputs,
        outputs,
        event_steps,
    })
}

#[must_use]
pub fn run_mode(mode: TrialMode, workload: &ValidatedWorkload) -> Vec<u64> {
    let program = trial_program();
    let initial_state = 0x8000_0000_0000_0043u64;
    match mode {
        TrialMode::OutputCopy => workload.outputs.clone(),
        TrialMode::Dense => program.forward_dense(&workload.inputs, 0, initial_state),
        TrialMode::Sparse => {
            program
                .forward_sparse(&workload.inputs, 0, initial_state)
                .expect("validated event stream always has increasing sample ticks")
                .outputs
        }
        TrialMode::EventScan => program.forward_event_scan(&workload.inputs, 0, initial_state),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrialResult {
    pub rounds: u32,
    /// Elapsed Rust monotonic time *inside* the measured operation. Includes
    /// vector allocation, full requested output production and checksum work.
    pub elapsed_nanoseconds: u128,
    /// Deterministic consume-proof; not a cryptographic output digest.
    pub checksum: u64,
}

pub fn run_timed(
    mode: TrialMode,
    workload: &ValidatedWorkload,
    rounds: u32,
) -> Result<TrialResult, TrialError> {
    if !(1..=1024).contains(&rounds) {
        return Err(TrialError::InvalidRepetitions);
    }
    let mut checksum = 0x8b8b_514f_1029_758du64;
    let start = Instant::now();
    for round in 0..rounds {
        let outputs = run_mode(mode, workload);
        // Read the complete result, not just an arbitrary first bit. The same
        // overhead is intentionally charged to every mode including copy.
        if outputs.len() != workload.inputs.len() {
            return Err(TrialError::OracleMismatch);
        }
        for &output in &outputs {
            checksum = checksum.rotate_left(1) ^ output;
        }
        checksum ^= u64::from(round).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        black_box(outputs);
    }
    Ok(TrialResult {
        rounds,
        elapsed_nanoseconds: start.elapsed().as_nanos(),
        checksum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_front_density_is_exact_even_at_one_hundred_percent() {
        for pct in TRIAL_FRONT_DENSITIES {
            for steps in [1usize, 37, 100, 257, 1024] {
                let inputs = exact_front_inputs(steps, pct).unwrap();
                assert_eq!(
                    count_rising_steps(&inputs, 0),
                    (steps as u64 * u64::from(pct)) / 100,
                    "density={pct}, length={steps}"
                );
            }
        }
        let full = exact_front_inputs(1024, 100).unwrap();
        assert_eq!(full.iter().filter(|&&word| word != 0).count(), 1024);
        assert_ne!(full[0], full[1]);
    }

    #[test]
    fn full_sparse_eventscan_and_reference_are_bit_exact() {
        for pct in TRIAL_FRONT_DENSITIES {
            for steps in [1usize, 63, 100, 257, 1024] {
                let workload = validated_workload(steps, pct).unwrap();
                for mode in [
                    TrialMode::OutputCopy,
                    TrialMode::Dense,
                    TrialMode::Sparse,
                    TrialMode::EventScan,
                ] {
                    assert_eq!(run_mode(mode, &workload), workload.outputs);
                }
            }
        }
    }

    #[test]
    fn timed_modes_consume_the_same_outputs_and_report_nonzero_work() {
        let workload = validated_workload(128, 50).unwrap();
        let copy = run_timed(TrialMode::OutputCopy, &workload, 3).unwrap();
        assert_eq!(copy.rounds, 3);
        let mut previous = copy.checksum;
        for mode in [TrialMode::Dense, TrialMode::Sparse, TrialMode::EventScan] {
            let trial = run_timed(mode, &workload, 3).unwrap();
            assert_eq!(trial.checksum, previous);
            previous = trial.checksum;
        }
        assert_ne!(copy.checksum, 0);
    }

    #[test]
    fn workload_identity_is_deterministic_and_sensitive_to_density() {
        let first = validated_workload(1024, 10).unwrap();
        let again = validated_workload(1024, 10).unwrap();
        let different = validated_workload(1024, 50).unwrap();
        assert_eq!(first, again);
        assert_ne!(first.fingerprint, different.fingerprint);
    }

    #[test]
    fn resource_caps_reject_infeasible_requested_trials() {
        assert_eq!(exact_front_inputs(0, 10), Err(TrialError::InvalidSteps));
        assert_eq!(
            exact_front_inputs(65_537, 10),
            Err(TrialError::InvalidSteps)
        );
        assert_eq!(
            exact_front_inputs(100, 101),
            Err(TrialError::InvalidDensity)
        );
        let valid = validated_workload(64, 10).unwrap();
        assert_eq!(
            run_timed(TrialMode::Dense, &valid, 0),
            Err(TrialError::InvalidRepetitions)
        );
        assert_eq!(
            run_timed(TrialMode::Dense, &valid, 1025),
            Err(TrialError::InvalidRepetitions)
        );
    }
}
