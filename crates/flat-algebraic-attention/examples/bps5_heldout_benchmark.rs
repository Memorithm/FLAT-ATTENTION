//! BPS-5 deterministic CSV report; research measurements, no physical energy.
//! The scenarios are synthetic teacher-generated tasks, not FP32 LSTM baselines.
use flat_algebraic_attention::boolean_benchmark_protocol::{
    compare_bps3, Bps5Protocol, Bps5Score,
};
use flat_algebraic_attention::boolean_coupled_learning::JointTrainPolicy;
use flat_algebraic_attention::boolean_edge_memory::BooleanEdgeProgram;
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;

fn emit(
    scenario: &str,
    occupancy: u8,
    eventful: u128,
    test_steps: usize,
    method: &str,
    score: Bps5Score,
) {
    println!(
        "{scenario},{occupancy},{eventful},{test_steps},{method},{},{},{},{},{},{},{},{},{},{},false",
        score.train_hamming,
        score.validation_hamming,
        score.test_hamming,
        score.test_bit_opportunities,
        score.training_candidate_evaluations,
        score.rotation,
        score.test_resources.sampled_words,
        score.test_resources.eventful_recurrent_steps,
        score.test_resources.emitted_output_words,
        score.test_resources.state_bit_flips
    );
}

fn main() {
    let keep = UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    };
    println!(
        "scenario,requested_occupancy_percent,actual_eventful_test_steps,test_steps,method,train_hamming,validation_hamming,test_hamming,test_bit_opportunities,training_candidate_evaluations,rotation,sampled_test_words,evaluated_test_recurrences,emitted_test_words,test_state_bit_flips,physical_energy_measured"
    );
    for (scenario, write_mask, pulse_lane) in [
        ("rotated_delayed_state", 0u64, Some(3)),
        ("conditional_edge_write", 1u64 << 3, Some(3)),
    ] {
        let teacher = BooleanEdgeProgram::new(
            keep,
            UnaryGate {
                when_false: 0,
                when_true: write_mask,
            },
            u64::MAX,
            7,
        )
        .expect("bounded teacher rotation");
        let student = BooleanEdgeProgram::new(
            keep,
            UnaryGate {
                when_false: 0,
                when_true: 1u64 << 1,
            },
            u64::MAX,
            0,
        )
        .expect("bounded student rotation");
        for occupancy in [0u8, 10, 50, 100] {
            let result = compare_bps3(
                Bps5Protocol {
                    seed: 0xbab5_5eed_0017_2026 ^ u64::from(occupancy),
                    train_episodes: 8,
                    validation_episodes: 4,
                    test_episodes: 8,
                    steps_per_episode: 32,
                    input_occupancy_percent: occupancy,
                    pulse_lane,
                    training: JointTrainPolicy {
                        max_epochs: 3,
                        max_candidate_evaluations: 2048,
                    },
                },
                teacher,
                student,
            )
            .expect("valid benchmark protocol");
            for (method, score) in [
                ("untrained", result.baseline_untrained),
                ("rotation_only", result.rotation_only),
                ("joint_gate_rotation", result.jointly_trained),
            ] {
                emit(
                    scenario,
                    occupancy,
                    result.actual_eventful_steps_in_test,
                    result.sample_steps_in_test,
                    method,
                    score,
                );
            }
        }
    }
}
