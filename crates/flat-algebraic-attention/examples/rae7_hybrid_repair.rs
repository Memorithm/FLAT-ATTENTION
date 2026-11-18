//! RAE-7: a dirty-word impulse proposal with bounded numerical repair.
//!
//! The printed mask is a research candidate, never a production skip permit.

use flat_algebraic_attention::rae_dirty_words::{DirtyFeatureWord, DirtyWordPageRouter};
use flat_algebraic_attention::rae_hybrid_repair::{
    plan_bounded_hybrid_attention, RaeHybridPlan,
};
use flat_algebraic_attention::rae_norm_certificate::{
    RaeBoundVersions, RaeCausalPageBounds,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // An exact external dirty-word producer reports a front on feature 0.
    let mut router = DirtyWordPageRouter::new(3, 3, 0, &[0], &[0, 1, 2])?;
    let mut wakes = Vec::new();
    let mut new_latches = Vec::new();
    router.apply_tick(
        1,
        &[DirtyFeatureWord { index: 0, next: 1 }],
        &mut wakes,
        &mut new_latches,
    )?;
    let impulse_mask = wakes.iter().fold(0_u64, |mask, page| mask | (1_u64 << *page));

    // Independently pinned per-page causal counts / max-K norms.
    // Only page 0's actual QK logit is supplied at planning time.
    let metadata = [
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
    let versions = RaeBoundVersions {
        query_key_generation: 1,
        norms_key_generation: 1,
        query_value_generation: 1,
        value_bound_generation: 1,
        query_mask_generation: 1,
        counts_mask_generation: 1,
    };
    let result = plan_bounded_hybrid_attention(
        &[impulse_mask],
        &metadata,
        &[8.0],
        1.0,
        1.0,
        versions,
        0.001,
        0.001,
        1,
        false,
    )?;
    match result {
        RaeHybridPlan::BoundedSparse {
            selected_page_mask,
            promoted_pages,
            certificate,
        } => {
            assert_eq!(selected_page_mask, vec![0b011]);
            assert_eq!(promoted_pages, vec![1]);
            println!(
                "RAE-7 research: wakes={wakes:?} repaired_mask={selected_page_mask:?} promoted={promoted_pages:?} max_dropped_mass={} max_output_error={}",
                certificate.maximum_dropped_mass,
                certificate.maximum_output_error
            );
        }
        RaeHybridPlan::DenseFallback { reason, .. } => {
            // The only permitted route on failure is original dense numerical attention.
            println!("RAE-7 research: dense fallback due to {reason:?}");
        }
    }
    Ok(())
}
