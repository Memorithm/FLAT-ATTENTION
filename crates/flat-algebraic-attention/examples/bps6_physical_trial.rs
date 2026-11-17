//! BPS-6 self-contained, bounded CPU measurement trial.
//!
//! Markers are intended for a *separate* trusted RAPL observer on RemoteOps.
//! The observed energy covers the whole physical host, including interference.
//! No energy sensor is accessed here; no claim of superiority is made.

use std::io::{self, Write};

use flat_algebraic_attention::boolean_physical_protocol::{
    run_timed, validated_workload, TrialError, TrialMode, TRIAL_FRONT_DENSITIES,
};

fn flush() {
    io::stdout().flush().expect("stdout marker flush");
}

fn run(smoke: bool) -> Result<(), TrialError> {
    let steps = if smoke { 100 } else { 4_096 };
    let rounds = if smoke { 1 } else { 128 };
    let repetitions = if smoke { 1 } else { 3 };
    let densities: &[u8] = if smoke {
        &[0, 50, 100]
    } else {
        &TRIAL_FRONT_DENSITIES
    };
    println!(
        "BPS6_PROTOCOL version=1 smoke={smoke} steps={steps} rounds={rounds} passes={repetitions} physical_energy_measured=false"
    );
    flush();

    let mut phase_count = 0usize;
    for &density in densities {
        let workload = validated_workload(steps, density)?;
        println!(
            "BPS6_CORPUS density={density} steps={steps} observed_front_steps={} input_fingerprint={:016x}",
            workload.event_steps, workload.fingerprint
        );
        flush();

        for pass in 0..repetitions {
            let normal = [
                TrialMode::OutputCopy,
                TrialMode::Dense,
                TrialMode::Sparse,
                TrialMode::EventScan,
            ];
            let reversed = [
                TrialMode::EventScan,
                TrialMode::Sparse,
                TrialMode::Dense,
                TrialMode::OutputCopy,
            ];
            let modes = if pass % 2 == 0 { normal } else { reversed };

            for mode in modes {
                let phase = format!("R{:02}_D{:03}_{}", pass + 1, density, mode.short_label());
                println!("BPS6_MARKER={phase}_BEGIN");
                flush();
                let trial = run_timed(mode, &workload, rounds)?;
                println!("BPS6_MARKER={phase}_END");
                flush();
                println!(
                    "BPS6_TRIAL phase={phase} density={density} mode={} steps={steps} rounds={} front_steps={} elapsed_ns={} checksum={:016x} energy_measured_by_binary=false",
                    mode.as_str(),
                    trial.rounds,
                    workload.event_steps,
                    trial.elapsed_nanoseconds,
                    trial.checksum
                );
                flush();
                phase_count += 1;
            }
        }
    }
    println!("BPS6_COMPLETE=true phases={phase_count} smoke={smoke}");
    flush();
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let smoke = match (args.next().as_deref(), args.next()) {
        (None, None) => false,
        (Some("--smoke"), None) => true,
        _ => panic!("usage: bps6_physical_trial [--smoke]"),
    };
    run(smoke).expect("BPS-6 exact-front and oracle qualification failed");
}
