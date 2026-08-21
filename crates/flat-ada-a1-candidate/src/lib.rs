//! Isolated ADA-A1 GPU research candidate for FLAT-ATTENTION.
//!
//! This crate is deliberately outside FLAT's production router. It exposes a
//! Q4 MHA shader that changes only the Online Softmax recurrence: after the
//! first admissible key, exactly one branch executes an explicit `exp` per
//! score update. Promotion requires independent GPU parity and hardware A/B
//! evidence; this crate by itself makes no production-performance claim.

#![forbid(unsafe_code)]

/// ADA-A1 Q4 portable MHA candidate shader.
pub const ADA_A1_FWD_WGSL: &str = include_str!("../shaders/flat_fwd_ada_a1.wgsl");

/// Query rows computed by one candidate workgroup; intentionally identical to
/// FLAT's qualified portable Q4 kernel.
pub const ADA_A1_QUERY_ROWS: usize = 4;

/// Logical scalar Online Softmax exp counts for one query with `admissible_keys`.
///
/// The baseline model is `2n-1`; ADA-A1 is `n-1`. These are algorithmic counts,
/// not claims about emitted GPU instructions or SFU utilization.
pub const fn logical_exp_counts(admissible_keys: usize) -> (usize, usize) {
    if admissible_keys == 0 {
        (0, 0)
    } else {
        (
            admissible_keys.saturating_mul(2).saturating_sub(1),
            admissible_keys.saturating_sub(1),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_counts_match_a1_contract() {
        assert_eq!(logical_exp_counts(0), (0, 0));
        assert_eq!(logical_exp_counts(1), (1, 0));
        assert_eq!(logical_exp_counts(128), (255, 127));
        assert_eq!(logical_exp_counts(4096), (8191, 4095));
    }
}
