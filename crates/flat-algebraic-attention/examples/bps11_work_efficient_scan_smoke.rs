//! BPS-11 exact work-efficient Boolean prefix-scan smoke (not a benchmark).
use flat_algebraic_attention::boolean_scan_lstm::{BooleanGateProgram, UnaryGate};

fn main() {
    let program = BooleanGateProgram {
        forget: UnaryGate {
            when_false: u64::MAX,
            when_true: 0x55aa_55aa_55aa_55aa,
        },
        write: UnaryGate {
            when_false: 0x0101_0001_0101_0001,
            when_true: 0xaa55_aa55_aa55_aa55,
        },
        output: UnaryGate {
            when_false: u64::MAX,
            when_true: 0xf0f0_0f0f_f0f0_0f0f,
        },
    };
    let initial = 0xb1e4_d8a9_7205_1c3fu64;
    for length in [0usize, 1, 3, 7, 64, 65, 129, 257, 1025] {
        let mut seed = 0x1c5a_9d07_24bb_6ea3u64;
        let inputs: Vec<u64> = (0..length)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            })
            .collect();
        let oracle = program.forward_serial(&inputs, initial);
        let work_efficient = program.forward_scan_work_efficient(&inputs, initial);
        let old_scan = program.forward_scan(&inputs, initial);
        assert_eq!(work_efficient, oracle, "Blelloch length={length}");
        assert_eq!(old_scan, oracle, "Hillis-Steele length={length}");
        println!("bps11 length={length} lanes=64 exact=true");
    }
    println!("bps11_result=PASS no_speedup_or_energy_claim");
}
