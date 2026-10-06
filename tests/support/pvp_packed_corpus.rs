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
        let terms = (0..gates)
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
}
