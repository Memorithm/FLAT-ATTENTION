//! Isolated ADA-A1 GPU research candidates for FLAT-ATTENTION.
//!
//! This crate is deliberately outside FLAT's production router. It preserves
//! the rejected branch-specialized Q4 realization as a negative control and
//! derives an adapted steady-state branchless realization by changing only the
//! Online Softmax state update. Promotion requires independent GPU parity and
//! hardware A/B evidence; this crate by itself makes no production-performance
//! claim.

#![forbid(unsafe_code)]

/// ADA-A1 branch-specialized Q4 portable MHA candidate shader.
///
/// This realization is retained as the frozen negative-control mapping after
/// its first physical-Thor timestamp smoke was slower than qualified Q4.
pub const ADA_A1_FWD_WGSL: &str = include_str!("../shaders/flat_fwd_ada_a1.wgsl");

/// Query rows computed by one candidate workgroup; intentionally identical to
/// FLAT's qualified portable Q4 kernel.
pub const ADA_A1_QUERY_ROWS: usize = 4;

const BRANCHED_STEADY_STATE: &str = r#"                        } else if (score <= previous_max) {
                            // Old maximum survives: only the new term needs exp.
                            p = exp(score - previous_max);
                            running_sum_shared[qr] = running_sum_shared[qr] + p;
                        } else {
                            // New maximum: rescale old state, while p = exp(0) = 1.
                            alpha = exp(previous_max - score);
                            running_max_shared[qr] = score;
                            running_sum_shared[qr] = running_sum_shared[qr] * alpha + 1.0;
                        }"#;

const BRANCHLESS_STEADY_STATE: &str = r#"                        } else {
                            // One exp for either branch, selected without dynamic control flow.
                            let delta = score - previous_max;
                            let e = exp(-abs(delta));
                            let score_is_new_max = score > previous_max;
                            alpha = select(1.0, e, score_is_new_max);
                            p = select(e, 1.0, score_is_new_max);
                            running_max_shared[qr] = max(previous_max, score);
                            running_sum_shared[qr] = running_sum_shared[qr] * alpha + p;
                        }"#;

/// Replace the first occurrence of `from` with `to`, failing loudly when the
/// anchor is missing.
///
/// The ADA-A1B shader is derived from the frozen A1 template by exact-text
/// surgery. A drifted anchor must abort instead of silently returning a
/// still-branched "branchless" source, which would invalidate benchmark
/// evidence gathered under the wrong recurrence.
fn replace_anchor(source: &str, from: &str, to: &str) -> String {
    let replaced = source.replacen(from, to, 1);
    assert!(
        replaced != source,
        "ADA-A1B template anchor not found or replacement is the identity: {from:?}"
    );
    replaced
}

/// Build the ADA-A1B steady-state branchless Q4 shader source.
///
/// The frozen A1 shader is used as the template so that geometry, bindings,
/// staging, barriers, reductions, output layout, and dispatch remain identical.
/// Only the branch between "old maximum" and "new maximum" is replaced by
/// `exp(-abs(delta))` plus `select`. The first admissible key still uses the
/// exact no-exp initialization branch, so the logical count remains `n - 1`.
pub fn ada_a1_branchless_wgsl() -> String {
    let source = replace_anchor(
        ADA_A1_FWD_WGSL,
        "fn flat_attention_forward_ada_a1(",
        "fn flat_attention_forward_ada_a1_branchless(",
    );
    let source = replace_anchor(
        &source,
        "executes exactly one branch containing one exp.",
        "executes exactly one exp in a branchless steady-state update.",
    );
    replace_anchor(&source, BRANCHED_STEADY_STATE, BRANCHLESS_STEADY_STATE)
}

/// Logical scalar Online Softmax exp counts for one query with `admissible_keys`.
///
/// The baseline model is `2n-1`; both ADA-A1 mappings are `n-1`. These are
/// algorithmic counts, not claims about emitted GPU instructions or SFU
/// utilization.
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
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    #[test]
    fn logical_counts_match_a1_contract() {
        assert_eq!(logical_exp_counts(0), (0, 0));
        assert_eq!(logical_exp_counts(1), (1, 0));
        assert_eq!(logical_exp_counts(128), (255, 127));
        assert_eq!(logical_exp_counts(4096), (8191, 4095));
    }

    #[test]
    fn branchless_source_changes_only_the_intended_recurrence_form() {
        let source = ada_a1_branchless_wgsl();
        assert!(source.contains("flat_attention_forward_ada_a1_branchless"));
        assert!(source.contains("let e = exp(-abs(delta));"));
        assert!(source.contains("alpha = select(1.0, e, score_is_new_max);"));
        assert!(source.contains("p = select(e, 1.0, score_is_new_max);"));
        assert!(!source.contains("else if (score <= previous_max)"));
        assert_ne!(source, ADA_A1_FWD_WGSL);

        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("ADA-A1B WGSL parse failed: {error:?}"));
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .unwrap_or_else(|error| panic!("ADA-A1B WGSL validation failed: {error:?}"));
    }
}
