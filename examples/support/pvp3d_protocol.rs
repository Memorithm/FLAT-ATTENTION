//! Pure protocol helpers shared by the executable and host contract tests.

pub const K_GRID: [usize; 7] = [256, 1024, 4096, 16_384, 65_536, 262_144, 1_048_576];
pub const G_GRID: [usize; 3] = [128, 512, 2048];
pub const SMOKE_GRID: [(usize, usize); 5] = [(1, 31), (4, 129), (8, 257), (64, 129), (256, 512)];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Candidate {
    Vec4 = 0,
    Fused2 = 1,
    Tile8 = 2,
}

impl Candidate {
    pub const ALL: [Self; 3] = [Self::Vec4, Self::Fused2, Self::Tile8];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Vec4 => "vec4",
            Self::Fused2 => "fused2",
            Self::Tile8 => "tile8",
        }
    }

    pub fn dispatches(self, k: usize) -> u32 {
        let stages = k.trailing_zeros();
        match self {
            Self::Fused2 if k >= 4 => stages - 1,
            Self::Tile8 if k >= 8 => stages - 2,
            _ => stages,
        }
    }
}

pub fn candidate_order(iteration: usize) -> [Candidate; 3] {
    use Candidate::{Fused2 as B, Tile8 as C, Vec4 as A};
    [
        [A, B, C],
        [A, C, B],
        [B, A, C],
        [B, C, A],
        [C, A, B],
        [C, B, A],
    ][iteration % 6]
}

pub fn source_revision_valid(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub fn percentile_ns(samples: &[u128], percentile: usize) -> u128 {
    assert!(!samples.is_empty() && (1..=100).contains(&percentile));
    let mut values = samples.to_vec();
    values.sort_unstable();
    values[(percentile * values.len()).div_ceil(100) - 1]
}

/// FNV-1a over canonical little-endian words; independent of host byte order.
pub fn checksum(words: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for word in words {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub buffer_bytes: u64,
    pub storage_binding_bytes: u64,
    pub workgroups: u32,
    pub invocations: u32,
    pub workgroup_x: u32,
    pub workgroup_storage_bytes: u32,
}

/// Admit the complete cohort before any GPU allocation. Unknown/overflowing
/// geometries are rejected rather than reduced or assigned a partial winner.
pub fn rejection(k: usize, g: usize, limits: Limits) -> Option<&'static str> {
    if k == 0 || !k.is_power_of_two() || g == 0 {
        return Some("invalid_geometry");
    }
    let Some(vectors) = g.checked_add(127).map(|value| value / 128) else {
        return Some("arithmetic_overflow");
    };
    let Some(words) = k
        .checked_mul(vectors)
        .and_then(|value| value.checked_mul(4))
    else {
        return Some("arithmetic_overflow");
    };
    if u32::try_from(k).is_err() || u32::try_from(words).is_err() {
        return Some("wgsl_u32_index_space");
    }
    let Some(bytes) = words
        .checked_mul(4)
        .and_then(|value| u64::try_from(value).ok())
    else {
        return Some("arithmetic_overflow");
    };
    if bytes > limits.buffer_bytes || bytes > limits.storage_binding_bytes {
        return Some("state_buffer_limit");
    }
    if limits.invocations < 64 || limits.workgroup_x < 64 {
        return Some("workgroup_size_limit");
    }
    if limits.workgroup_storage_bytes < 128 {
        return Some("tile8_workgroup_storage_limit");
    }
    let vec4_groups = (k / 2 * vectors).div_ceil(64);
    let fused_groups = if k >= 4 {
        (k / 4 * vectors).div_ceil(64)
    } else {
        vec4_groups
    };
    let tile8_groups = if k >= 8 { k / 8 * vectors } else { vec4_groups };
    if [vec4_groups, fused_groups, tile8_groups]
        .into_iter()
        .any(|groups| groups > limits.workgroups as usize)
    {
        return Some("cohort_dispatch_limit");
    }
    None
}
