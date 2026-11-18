//! BPS-22: deterministic neuron-by-neuron Boolean input activation,
//! per-operator gate/source routing profile and bit-exact recurrent output.
use flat_algebraic_attention::boolean_neuron_activity::{
    exact_weight_corpus, NeuronActivationProfile,
};
use flat_algebraic_attention::boolean_neuron_graph::{
    AdaptiveRoutingPolicy, BooleanNeuronGraph, NEURON_COUNT, OPERATOR_COUNT,
};
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;

fn main() {
    let mut shifts = [[0u8; NEURON_COUNT]; OPERATOR_COUNT];
    for lane in 0..NEURON_COUNT {
        shifts[0][lane] = lane as u8;
        shifts[1][lane] = 3;
        shifts[2][lane] = ((lane * 5) % NEURON_COUNT) as u8;
        shifts[3][lane] = ((lane * 7) % NEURON_COUNT) as u8;
    }
    let all = UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    };
    let graph = BooleanNeuronGraph::new(
        [
            UnaryGate {
                when_false: 0,
                when_true: u64::MAX,
            },
            all,
            UnaryGate {
                when_false: u64::MAX,
                when_true: 0,
            },
            UnaryGate {
                when_false: 0x0000_0000_0000_f00f,
                when_true: 0xf000_1000_00ff_0011,
            },
        ],
        UnaryGate::default(),
        all,
        shifts,
    )
    .unwrap();

    let policy = AdaptiveRoutingPolicy::default();
    let mut sampled = 0u64;
    let mut observed_distinct_weights = 0u64;
    for weight in [0, 1, 4, 16, 32, 48, 63, 64] {
        let corpus = exact_weight_corpus(0xb221_425e_b887_6300, weight, 127).unwrap();
        assert!(corpus
            .iter()
            .all(|word| word.count_ones() == u32::from(weight)));
        let profile = NeuronActivationProfile::from_inputs(&graph, &corpus, policy).unwrap();
        assert_eq!(profile.samples, 127);
        assert_eq!(profile.input_one_bits, 127 * u64::from(weight));
        assert_eq!(profile.input_weight_histogram[usize::from(weight)], 127);
        for counts in profile.candidate_routes {
            assert_eq!(counts.total(), Some(127));
        }
        let initial = 0x1f41_97ab_2811_340du64;
        let expected = graph.forward_scalar(&corpus, initial);
        assert_eq!(
            graph.forward_activity_hybrid(&corpus, initial, policy),
            expected
        );
        assert_eq!(graph.forward_adaptive(&corpus, initial, policy), expected);
        sampled += 127;
        observed_distinct_weights += 1;
    }
    assert_eq!(sampled, 1016);
    assert_eq!(observed_distinct_weights, 8);
    assert_eq!(
        exact_weight_corpus(9, 65, 1).unwrap_err(),
        flat_algebraic_attention::boolean_neuron_activity::LaneMaskError::WeightTooLarge {
            requested: 65
        }
    );
    println!("BPS22_MASK_WEIGHT_BUCKETS=8");
    println!("BPS22_SAMPLES=1016");
    println!("BPS22_PER_NEURON_ACTIVATION=true");
    println!("BPS22_SCALAR_OUTPUT_EXACT=true");
    println!("BPS22_COMPLETE=true");
}
