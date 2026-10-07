//! Deterministic independent ANF fixtures for the separate packed candidate.

pub const GEOMETRIES: [(usize, usize); 13] = [
    (1, 1),
    (2, 31),
    (4, 65),
    (8, 129),
    (16, 3),
    (32, 5),
    (64, 257),
    (128, 129),
    (256, 513),
    (512, 7),
    (1024, 33),
    (2048, 257),
    (256, 2048),
];
pub const BANKS: [&str; 3] = ["boundary", "sparse4", "dense32"];

pub struct AnfBank {
    k: usize,
    terms: Vec<Vec<usize>>,
}

impl AnfBank {
    pub fn frozen(k: usize, gates: usize, kind: &str) -> Self {
        Self::frozen_for_round(k, gates, kind, 0)
    }

    /// Round zero preserves the original fixture; later rounds toggle one
    /// high-variable coefficient on alternating gates to expose stale states.
    pub fn frozen_for_round(k: usize, gates: usize, kind: &str, round: usize) -> Self {
        assert!(k.is_power_of_two());
        assert!(round < 3);
        let mut terms: Vec<Vec<usize>> = (0..gates)
            .map(|gate| match kind {
                "boundary" => match gate % 6 {
                    0 => vec![],
                    1 => vec![0],
                    2 => vec![usize::from(k > 1)],
                    3 => vec![k - 1],
                    4 => vec![0, usize::from(k > 1)],
                    _ => (0..k.ilog2()).map(|bit| 1_usize << bit).collect(),
                },
                "sparse4" | "dense32" => {
                    let count = k.min(if kind == "sparse4" { 4 } else { 32 });
                    (0..count)
                        .map(|term| (257 * gate + 73 * term) % k)
                        .collect()
                }
                _ => panic!("unknown packed ANF fixture"),
            })
            .collect();
        if round > 0 {
            let mask = k >> round;
            for (gate, gate_terms) in terms.iter_mut().enumerate() {
                if gate % 2 == (round - 1) % 2 {
                    gate_terms.push(mask);
                }
            }
        }
        Self { k, terms }
    }

    pub fn coefficients(&self) -> Vec<u64> {
        let row_words = self.k.div_ceil(64);
        let mut output = vec![0; row_words * self.terms.len()];
        for (gate, terms) in self.terms.iter().enumerate() {
            for &mask in terms {
                output[gate * row_words + mask / 64] ^= 1_u64 << (mask % 64);
            }
        }
        output
    }

    /// Direct monomial evaluation; no bitplane adapter or transform is used.
    pub fn truth_u32_words(&self) -> Vec<u32> {
        let row_words = self.k.div_ceil(128) * 4;
        let mut output = vec![0; row_words * self.terms.len()];
        for (gate, terms) in self.terms.iter().enumerate() {
            for address in 0..self.k {
                let truth = terms
                    .iter()
                    .fold(false, |parity, &mask| parity ^ ((address & mask) == mask));
                if truth {
                    output[gate * row_words + address / 32] |= 1_u32 << (address % 32);
                }
            }
        }
        output
    }

    /// Direct monomial evaluation in 32-address words, not a zeta network.
    /// For a monomial mask m and address 32*w+b, the predicate factors into
    /// (w & (m >> 5)) == (m >> 5) and (b & (m & 31)) == (m & 31).
    pub fn truth_u32_words_by_monomial_masks(&self) -> Vec<u32> {
        let mut low_truth = [0_u32; 32];
        for (mask, pattern) in low_truth.iter_mut().enumerate() {
            for bit in 0..32 {
                if bit & mask == mask {
                    *pattern |= 1_u32 << bit;
                }
            }
        }
        let row_words = self.k.div_ceil(128) * 4;
        let logical_words = self.k.div_ceil(32);
        let active_mask = if self.k < 32 {
            (1_u32 << self.k) - 1
        } else {
            u32::MAX
        };
        let mut output = vec![0; row_words * self.terms.len()];
        for (gate, terms) in self.terms.iter().enumerate() {
            for word in 0..logical_words {
                let mut truth = 0;
                for &mask in terms {
                    let high = mask >> 5;
                    if word & high == high {
                        truth ^= low_truth[mask & 31];
                    }
                }
                output[gate * row_words + word] = truth & active_mask;
            }
        }
        output
    }
}
