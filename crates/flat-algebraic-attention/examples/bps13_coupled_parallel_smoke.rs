//! BPS-13 smoke: exact coupled Boolean recurrence, not a speed benchmark.
use flat_algebraic_attention::boolean_edge_memory::BooleanEdgeProgram;
use flat_algebraic_attention::boolean_rotating_memory::BooleanRotatingProgram;
use flat_algebraic_attention::boolean_scan_lstm::{BooleanGateProgram, UnaryGate};

fn main() {
    let keep = UnaryGate {
        when_false: u64::MAX,
        when_true: 0x55aa_55aa_55aa_55aa,
    };
    let write = UnaryGate {
        when_false: 0,
        when_true: 0x0140_0000_0010_0001,
    };
    let output = UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    };
    let gate = BooleanGateProgram {
        forget: keep,
        write,
        output,
    };
    let rotating = BooleanRotatingProgram::new(gate, 7).unwrap();
    let event = BooleanEdgeProgram::new(keep, write, u64::MAX, 7).unwrap();
    let mut seed = 0x8a1c_744f_41d9_2ea7u64;
    for length in [0usize, 1, 3, 65, 129, 2049, 4097] {
        let inputs: Vec<u64> = (0..length)
            .map(|tick| {
                if tick % 7 == 0 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed
                } else if tick % 17 == 0 {
                    0
                } else {
                    seed
                }
            })
            .collect();
        let baseline = 0xa19f_60bd_8881_4f1au64;
        let initial = 0xa15c_9b60_41ac_0101u64;
        for workers in [0, 1, 2, 8, 64] {
            assert_eq!(
                rotating.forward_scan_parallel(&inputs, initial, workers),
                rotating.forward_serial(&inputs, initial)
            );
            assert_eq!(
                event.forward_scan_parallel(&inputs, baseline, initial, workers),
                event.forward_dense(&inputs, baseline, initial)
            );
            assert_eq!(
                event.forward_event_scan_parallel(&inputs, baseline, initial, workers),
                event.forward_event_scan(&inputs, baseline, initial)
            );
        }
        println!("bps13 steps={length} exact_cross_lane=true exact_event=true");
    }
    println!("bps13_result=PASS no_physical_speed_or_energy_claim");
}
