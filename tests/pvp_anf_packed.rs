#[path = "support/pvp_packed_corpus.rs"]
mod corpus;

use flat_attention::pvp_packed::{
    pvp_subset_zeta_packed_host_in_place, FlatPvpPackedBitplanesV1, FlatPvpPackedLayoutV1,
};
use flat_attention::pvp_vec4::{
    pvp_subset_zeta_vec4_host_in_place, FlatPvpVec4BitplanesV1, FlatPvpVec4LayoutV1,
};

#[test]
fn packed_anf_truth_padding_inverse_and_logical_vec4_equivalence() {
    let mut cases = 0;
    for (k, g) in corpus::GEOMETRIES {
        for kind in corpus::BANKS {
            let bank = corpus::AnfBank::frozen(k, g, kind);
            let coefficients = bank.coefficients();
            let layout = FlatPvpPackedLayoutV1::new(k, g).unwrap();
            let original =
                FlatPvpPackedBitplanesV1::from_gate_major_u64(layout, &coefficients).unwrap();
            let mut actual = original.clone();
            let stats = pvp_subset_zeta_packed_host_in_place(&mut actual).unwrap();
            assert_eq!(
                actual.words(),
                bank.truth_u32_words(),
                "K={k} G={g} bank={kind}"
            );
            actual.validate_padding_zero().unwrap();
            assert_eq!(
                stats.logical_gate_xor_ops,
                (k / 2) as u128 * u128::from(k.ilog2()) * g as u128
            );
            assert_eq!(stats.scratch_u32_words, 0);

            let old_layout = FlatPvpVec4LayoutV1::new(k, g).unwrap();
            let mut old =
                FlatPvpVec4BitplanesV1::from_gate_major_u64(old_layout, &coefficients).unwrap();
            pvp_subset_zeta_vec4_host_in_place(&mut old).unwrap();
            for gate in 0..g {
                for address in 0..k {
                    assert_eq!(
                        actual.get(address, gate),
                        old.get(address, gate),
                        "layout equivalence K={k} G={g} gate={gate} address={address}"
                    );
                }
            }

            pvp_subset_zeta_packed_host_in_place(&mut actual).unwrap();
            assert_eq!(actual, original, "inverse K={k} G={g} bank={kind}");
            assert_eq!(actual.to_gate_major_u64().unwrap(), coefficients);
            cases += 1;
        }
    }
    assert_eq!(cases, 39);
}

#[test]
fn independent_boundary_oracle_has_known_truths() {
    let bank = corpus::AnfBank::frozen(8, 6, "boundary");
    let words = bank.truth_u32_words();
    let decoded: Vec<_> = (0..8)
        .map(|address| {
            (0..6).fold(0_u32, |row, gate| {
                row | (((words[gate * 4] >> address) & 1) << gate)
            })
        })
        .collect();
    assert_eq!(decoded, vec![18, 38, 50, 6, 50, 6, 18, 46]);
}
