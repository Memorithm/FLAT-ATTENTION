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

#[test]
fn wordwise_monomials_match_addresswise_oracle_including_round_nonces() {
    let mut cases = 0;
    for (k, g) in corpus::GEOMETRIES {
        for kind in corpus::BANKS {
            for round in 0..3 {
                let bank = corpus::AnfBank::frozen_for_round(k, g, kind, round);
                assert_eq!(
                    bank.truth_u32_words_by_monomial_masks(),
                    bank.truth_u32_words(),
                    "wordwise oracle K={k} G={g} bank={kind} round={round}"
                );
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 117);
}

#[test]
fn large_wordwise_boundary_truths_cover_high_variables() {
    for k in [16384_usize, 65536, 262144] {
        let bank = corpus::AnfBank::frozen(k, 6, "boundary");
        let words = bank.truth_u32_words_by_monomial_masks();
        let row_words = k / 32;
        for address in [0, 1, 31, 32, 63, 64, 127, 128, k / 2, k - 1] {
            let known = [
                false,
                true,
                address & 1 == 1,
                address == k - 1,
                address & 1 == 0,
                address.count_ones() % 2 == 1,
            ];
            for (gate, expected) in known.into_iter().enumerate() {
                let observed = (words[gate * row_words + address / 32] >> (address % 32)) & 1 == 1;
                assert_eq!(observed, expected, "K={k} gate={gate} address={address}");
            }
        }
    }
}
