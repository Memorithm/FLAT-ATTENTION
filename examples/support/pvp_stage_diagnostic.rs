//! Fixed, untimed transfer/transform diagnosis; not a performance protocol.

use flat_attention::pvp_vec4::{FlatPvpVec4BitplanesV1, FlatPvpVec4LayoutV1};

pub const ROUNDS: usize = 12;
pub const GEOMETRIES: [(usize, usize); 2] = [(65_536, 512), (262_144, 128)];
pub const FIRST_MISMATCH_LIMIT: usize = 8;

/// Identical xorshift fixture and canonical padding to the historical PVP3d banc.
pub fn fixture(layout: FlatPvpVec4LayoutV1) -> FlatPvpVec4BitplanesV1 {
    let mut words = vec![0_u32; layout.storage_u32_words()];
    let row_words = layout.vectors_per_address() * 4;
    let mut state = 0x9e37_79b9_u32 ^ layout.addresses() as u32 ^ layout.gates() as u32;
    for (index, word) in words.iter_mut().enumerate() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let remaining = layout.gates().saturating_sub(index % row_words * 32);
        *word = match remaining {
            0 => 0,
            1..=31 => state & ((1_u32 << remaining) - 1),
            _ => state,
        };
    }
    FlatPvpVec4BitplanesV1::from_words(layout, words).expect("canonical diagnostic fixture")
}

#[derive(Debug, PartialEq, Eq)]
pub struct Differences {
    pub count: usize,
    pub first: Vec<(usize, u32, u32)>,
}

/// Check every word, with bounded reporting independent of mismatch count.
pub fn differences(actual: &[u32], expected: &[u32]) -> Differences {
    assert_eq!(
        actual.len(),
        expected.len(),
        "diagnostic word count mismatch"
    );
    let mismatches = actual
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (actual, expected))| actual != expected);
    let count = mismatches.clone().count();
    let first = mismatches
        .take(FIRST_MISMATCH_LIMIT)
        .map(|(index, (actual, expected))| (index, *actual, *expected))
        .collect();
    Differences { count, first }
}
