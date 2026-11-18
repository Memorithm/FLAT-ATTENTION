//! RAE-8: executable causal scalar softmax over repaired impulse pages.
//! The example is research-only and claims no physical bandwidth advantage.

use flat_algebraic_attention::rae_hybrid_execution::{
    execute_rae_hybrid, RaeNumericPolicy, RaeNumericQuery, RaeNumericalRoute,
};
use flat_algebraic_attention::rae_norm_certificate::{
    RaeBoundVersions, RaeCausalPageBounds,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One original causal query with three 1D K/V rows. The event router
    // initially wakes only page 0. RAE-7 promotes page 1 before executing QK/V.
    let query = RaeNumericQuery {
        query: &[1.0],
        keys: &[8.0, -8.0, 0.0],
        values: &[1.0, -1.0, 0.5],
        value_width: 1,
        key_pages: &[0, 1, 2],
        eligible: &[true, true, true],
        scale: 1.0,
    };
    let pages = [
        RaeCausalPageBounds {
            eligible_keys: 1,
            max_key_l2_norm: 8.0,
        },
        RaeCausalPageBounds {
            eligible_keys: 1,
            max_key_l2_norm: 8.0,
        },
        RaeCausalPageBounds {
            eligible_keys: 1,
            max_key_l2_norm: 0.0,
        },
    ];
    let mut policy = RaeNumericPolicy {
        impulse_pages: &[0b001],
        bounds: &pages,
        versions: RaeBoundVersions {
            query_key_generation: 1,
            norms_key_generation: 1,
            query_value_generation: 1,
            value_bound_generation: 1,
            query_mask_generation: 1,
            counts_mask_generation: 1,
        },
        value_abs_bound: 1.0,
        maximum_dropped_mass: 0.001,
        maximum_output_error: 0.001,
        maximum_promotions: 1,
        context_invalidated: false,
    };
    let sparse = execute_rae_hybrid(query, policy)?;
    policy.context_invalidated = true;
    let dense = execute_rae_hybrid(query, policy)?;
    let RaeNumericalRoute::BoundedSparse {
        selected_page_mask,
        promoted_pages,
        certificate,
    } = &sparse.route else {
        panic!("expected a certified research sparse candidate");
    };
    assert_eq!(selected_page_mask, &[0b011]);
    assert_eq!(promoted_pages, &[1]);
    assert_eq!(sparse.key_rows_scored, 2);
    assert_eq!(dense.key_rows_scored, 3);
    let diff = (sparse.output[0] - dense.output[0]).abs();
    assert!(diff <= certificate.maximum_output_error);
    println!(
        "rae8_result=PASS sparse_keys={} dense_keys={} output_difference={} certified_error_bound={}",
        sparse.key_rows_scored,
        dense.key_rows_scored,
        diff,
        certificate.maximum_output_error
    );
    Ok(())
}
