//! BPS-12: bounded CPU worker scan compared against two exact serial oracles.
use flat_algebraic_attention::boolean_scan_lstm::{BooleanGateProgram, UnaryGate};

fn main() {
    let program = BooleanGateProgram {
        forget: UnaryGate {
            when_false: u64::MAX,
            when_true: 0x1818_1818_1818_1818,
        },
        write: UnaryGate {
            when_false: 0,
            when_true: u64::MAX,
        },
        output: UnaryGate {
            when_false: 0x7fff_ffff_ffff_ffff,
            when_true: u64::MAX,
        },
    };
    let mut seed = 0x4c81_6a30_9e57_b25du64;
    let inputs: Vec<u64> = (0..8193)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        })
        .collect();
    let initial = 1u64 << 63;
    let exact = program.forward_serial(&inputs, initial);
    assert_eq!(exact, program.forward_scan_work_efficient(&inputs, initial));
    for workers in [0usize, 1, 2, 4, 8, 64] {
        assert_eq!(
            exact,
            program.forward_scan_parallel(&inputs, initial, workers),
            "workers={workers}"
        );
        println!("bps12 exact=true lanes=64 length=8193 requested_workers={workers}");
    }
    println!("bps12_result=PASS no_latency_or_energy_claim");
}
