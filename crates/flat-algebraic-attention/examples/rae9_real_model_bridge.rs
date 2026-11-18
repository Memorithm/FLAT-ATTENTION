//! RAE-9: portable real-model tensor-capture adapter for GPT-2 XL and
//! SmolLM2-135M-Instruct, using exactly the post-transform Q/K/V fed to SDPA.
//!
//! Input is an ephemeral, bounded ASCII f32 snapshot created in the controlled
//! RemoteOps model forward. This does NOT load weights, change a production
//! backend, or establish physical K/V savings. Snapshot metadata is recomputed
//! from the captured vectors (full-snapshot scanning cost is NOT free).
#![forbid(unsafe_code)]

use flat_algebraic_attention::rae_dirty_words::{DirtyFeatureWord, DirtyWordPageRouter};
use flat_algebraic_attention::rae_hybrid_execution::{
    execute_rae_hybrid, RaeNumericPolicy, RaeNumericQuery, RaeNumericalRoute,
};
use flat_algebraic_attention::rae_norm_certificate::{RaeBoundVersions, RaeCausalPageBounds};
use std::env;
use std::fs;

const MAX_INPUT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ROWS: usize = 4_096;
const MAX_DIM: usize = 256;
const MAX_PAGES: usize = 512;

#[derive(Debug)]
struct Snapshot {
    model: String,
    rows: usize,
    dim: usize,
    vdim: usize,
    page_size: usize,
    scale: f64,
    generation: u64,
    query_heads: usize,
    kv_heads: usize,
    head: usize,
    kv_head: usize,
    previous_query: Vec<f64>,
    query: Vec<f64>,
    keys: Vec<f64>,
    values: Vec<f64>,
}

fn token<T: std::str::FromStr>(
    words: &mut std::str::SplitWhitespace<'_>,
    field: &str,
) -> Result<T, String> {
    let s = words
        .next()
        .ok_or_else(|| format!("missing field {field}"))?;
    s.parse().map_err(|_| format!("invalid field {field}"))
}

fn floats(
    words: &mut std::str::SplitWhitespace<'_>,
    len: usize,
    field: &str,
) -> Result<Vec<f64>, String> {
    (0..len)
        .map(|_| {
            let value: f64 = token(words, field)?;
            if !value.is_finite() {
                return Err(format!("non-finite {field}"));
            }
            Ok(value)
        })
        .collect()
}

fn parse_capture(data: &str) -> Result<Snapshot, String> {
    let mut words = data.split_whitespace();
    let magic: String = token(&mut words, "magic")?;
    if magic != "RAE9QKV1" {
        return Err("unsupported snapshot protocol".into());
    }
    let model: String = token(&mut words, "model")?;
    if model != "gpt2-xl" && model != "smollm2-135m-instruct" {
        return Err("unsupported model identity".into());
    }
    let rows: usize = token(&mut words, "rows")?;
    let dim: usize = token(&mut words, "dim")?;
    let vdim: usize = token(&mut words, "value_dim")?;
    let page_size: usize = token(&mut words, "page_size")?;
    let scale: f64 = token(&mut words, "scale")?;
    let generation: u64 = token(&mut words, "generation")?;
    let query_heads: usize = token(&mut words, "query_heads")?;
    let kv_heads: usize = token(&mut words, "kv_heads")?;
    let head: usize = token(&mut words, "head")?;
    let kv_head: usize = token(&mut words, "kv_head")?;
    if rows == 0
        || rows > MAX_ROWS
        || dim == 0
        || dim > MAX_DIM
        || vdim == 0
        || vdim > MAX_DIM
        || page_size == 0
        || page_size > rows
        || rows.div_ceil(page_size) > MAX_PAGES
        || !scale.is_finite()
        || scale < 0.0
        || generation == 0
        || generation == u64::MAX
        || query_heads == 0
        || kv_heads == 0
        || query_heads % kv_heads != 0
        || head >= query_heads
        || kv_head >= kv_heads
        || kv_head != head / (query_heads / kv_heads)
    {
        return Err("invalid shape, generation or grouped-head mapping".into());
    }
    if model == "gpt2-xl" && query_heads != kv_heads {
        return Err("GPT-2 XL cannot have grouped KV heads".into());
    }
    let previous_query = floats(&mut words, dim, "previous_q")?;
    let query = floats(&mut words, dim, "query")?;
    let keys = floats(&mut words, rows * dim, "keys")?;
    let values = floats(&mut words, rows * vdim, "values")?;
    if words.next().is_some() {
        return Err("trailing snapshot data".into());
    }
    Ok(Snapshot {
        model,
        rows,
        dim,
        vdim,
        page_size,
        scale,
        generation,
        query_heads,
        kv_heads,
        head,
        kv_head,
        previous_query,
        query,
        keys,
        values,
    })
}

fn packed_signs(vector: &[f64]) -> Vec<u64> {
    let mut words = vec![0_u64; vector.len().div_ceil(64)];
    for (feature, &value) in vector.iter().enumerate() {
        if value.is_sign_positive() && value != 0.0 {
            words[feature / 64] |= 1_u64 << (feature % 64);
        }
    }
    words
}

#[derive(Debug)]
struct Report {
    snapshot_model: String,
    head: usize,
    kv_head: usize,
    query_heads: usize,
    kv_heads: usize,
    original_keys: usize,
    initial_pages: usize,
    final_keys: usize,
    key_rows_scored: usize,
    final_pages: usize,
    route: String,
    maximum_output_error: f64,
    measured_dense_difference: f64,
    output: Vec<f64>,
}

fn evaluate(snapshot: &Snapshot) -> Result<Report, String> {
    let pages = snapshot.rows.div_ceil(snapshot.page_size);
    let key_pages: Vec<usize> = (0..snapshot.rows)
        .map(|index| index / snapshot.page_size)
        .collect();
    let eligible = vec![true; snapshot.rows];
    let mut metadata = vec![
        RaeCausalPageBounds {
            eligible_keys: 0,
            max_key_l2_norm: 0.0,
        };
        pages
    ];
    let mut value_abs_bound = 0.0_f64;
    // This full materialization is currently an explicit research overhead.
    for key in 0..snapshot.rows {
        let p = key_pages[key];
        metadata[p].eligible_keys += 1;
        let base = key * snapshot.dim;
        let norm = snapshot.keys[base..base + snapshot.dim]
            .iter()
            .copied()
            .fold(0.0_f64, f64::hypot);
        if !norm.is_finite() {
            return Err("nonfinite key norm".into());
        }
        metadata[p].max_key_l2_norm = metadata[p]
            .max_key_l2_norm
            .max(norm * (1.0 + 1e-12) + 1e-12);
        let vb = key * snapshot.vdim;
        for &value in &snapshot.values[vb..vb + snapshot.vdim] {
            value_abs_bound = value_abs_bound.max(value.abs());
        }
    }
    if !value_abs_bound.is_finite() {
        return Err("invalid value bound".into());
    }
    value_abs_bound = value_abs_bound * (1.0 + 1e-12) + 1e-12;

    // RAE-5 consumes genuine sign-bit changes of successive model queries.
    // This feature-to-page mapping (coordinate modulo page count) is a
    // deliberately untrained control, NOT a learned relevance map.
    let mapping: Vec<usize> = (0..snapshot.dim).map(|i| i % pages).collect();
    let previous = packed_signs(&snapshot.previous_query);
    let current = packed_signs(&snapshot.query);
    let mut router = DirtyWordPageRouter::new(
        snapshot.dim,
        pages,
        snapshot.generation,
        &previous,
        &mapping,
    )
    .map_err(|e| e.to_string())?;
    let dirty: Vec<DirtyFeatureWord> = previous
        .iter()
        .zip(&current)
        .enumerate()
        .filter_map(|(index, (&old, &next))| {
            (old != next).then_some(DirtyFeatureWord { index, next })
        })
        .collect();
    let mut wakes = Vec::new();
    let mut newly_latched = Vec::new();
    router
        .apply_tick(
            snapshot.generation + 1,
            &dirty,
            &mut wakes,
            &mut newly_latched,
        )
        .map_err(|e| e.to_string())?;
    let mut impulse_pages = vec![0_u64; pages.div_ceil(64)];
    for &page in &wakes {
        impulse_pages[page / 64] |= 1_u64 << (page % 64);
    }
    let v = RaeBoundVersions {
        query_key_generation: snapshot.generation,
        norms_key_generation: snapshot.generation,
        query_value_generation: snapshot.generation,
        value_bound_generation: snapshot.generation,
        query_mask_generation: snapshot.generation,
        counts_mask_generation: snapshot.generation,
    };
    let query = RaeNumericQuery {
        query: &snapshot.query,
        keys: &snapshot.keys,
        values: &snapshot.values,
        value_width: snapshot.vdim,
        key_pages: &key_pages,
        eligible: &eligible,
        scale: snapshot.scale,
    };
    let policy = RaeNumericPolicy {
        impulse_pages: &impulse_pages,
        bounds: &metadata,
        versions: v,
        value_abs_bound,
        maximum_dropped_mass: 0.001,
        maximum_output_error: 0.01,
        maximum_promotions: pages.saturating_sub(1),
        context_invalidated: false,
    };
    let repaired = execute_rae_hybrid(query, policy).map_err(|e| e.to_string())?;
    let dense = execute_rae_hybrid(
        query,
        RaeNumericPolicy {
            context_invalidated: true,
            ..policy
        },
    )
    .map_err(|e| e.to_string())?;
    let difference = repaired
        .output
        .iter()
        .zip(&dense.output)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let (route, bound, final_pages) = match repaired.route {
        RaeNumericalRoute::BoundedSparse {
            selected_page_mask,
            certificate,
            ..
        } => {
            if difference > certificate.maximum_output_error + 1e-12 {
                return Err("observed error exceeds the research certificate".into());
            }
            (
                "bounded_sparse".to_owned(),
                certificate.maximum_output_error,
                selected_page_mask
                    .iter()
                    .map(|word| word.count_ones() as usize)
                    .sum(),
            )
        }
        RaeNumericalRoute::DenseFallback { reason } => {
            if difference != 0.0 {
                return Err("dense fallback must be identical to dense oracle".into());
            }
            (format!("dense_{reason:?}"), 0.0, pages)
        }
    };
    Ok(Report {
        snapshot_model: snapshot.model.clone(),
        head: snapshot.head,
        kv_head: snapshot.kv_head,
        query_heads: snapshot.query_heads,
        kv_heads: snapshot.kv_heads,
        original_keys: snapshot.rows,
        initial_pages: wakes.len(),
        final_keys: repaired.executed_keys,
        key_rows_scored: repaired.key_rows_scored,
        final_pages,
        route,
        maximum_output_error: bound,
        measured_dense_difference: difference,
        output: repaired.output,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: rae9_real_model_bridge <snapshot.txt>")?;
    if args.next().is_some() {
        return Err("unexpected additional argument".into());
    }
    let file = fs::File::open(&path)?;
    if file.metadata()?.len() > MAX_INPUT_BYTES {
        return Err("snapshot size exceeds research limit".into());
    }
    let capture = fs::read_to_string(path)?;
    // The file is read from the same validated path, not a model cache.
    let snapshot = parse_capture(&capture)?;
    let report = evaluate(&snapshot)?;
    println!(
        "RAE9_RESULT model={} head={} kv_head={} q_heads={} kv_heads={} \
         keys={} initial_pages={} executed_keys={} scored_keys={} pages={} \
         route={} dense_abs_difference={:.12e} max_error_bound={:.12e} \
         physical_bandwidth_measured=false trained_route=false",
        report.snapshot_model,
        report.head,
        report.kv_head,
        report.query_heads,
        report.kv_heads,
        report.original_keys,
        report.initial_pages,
        report.final_keys,
        report.key_rows_scored,
        report.final_pages,
        report.route,
        report.measured_dense_difference,
        report.maximum_output_error
    );
    print!("RAE9_OUTPUT");
    for value in report.output {
        print!(" {:.17e}", value);
    }
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(model: &str, qh: usize, kvh: usize, head: usize, kv: usize) -> String {
        format!(
            "RAE9QKV1 {model} 3 1 1 1 1 7 {qh} {kvh} {head} {kv}\n\
             -1\n1\n8 -8 0\n1 -1 0.5"
        )
    }

    #[test]
    fn gpt2_xl_snapshot_follows_rising_impulse_and_matches_dense() {
        let parsed = parse_capture(&fixture("gpt2-xl", 4, 4, 2, 2)).unwrap();
        let report = evaluate(&parsed).unwrap();
        assert_eq!(report.initial_pages, 1);
        assert_eq!(report.final_keys, 2);
        assert_eq!(report.final_pages, 2);
        assert_eq!(report.route, "bounded_sparse");
        assert!(report.measured_dense_difference <= report.maximum_output_error);
    }

    #[test]
    fn smollm2_grouped_head_is_resolved_without_reindexing_original_keys() {
        let parsed = parse_capture(&fixture("smollm2-135m-instruct", 9, 3, 4, 1)).unwrap();
        let report = evaluate(&parsed).unwrap();
        assert_eq!(report.query_heads, 9);
        assert_eq!(report.kv_heads, 3);
        assert_eq!(report.kv_head, 1);
        assert_eq!(report.final_keys, 2);
    }

    #[test]
    fn missing_rising_events_fail_closed_to_full_dense() {
        let input = fixture("gpt2-xl", 4, 4, 2, 2).replace("-1\n1\n", "1\n1\n");
        let report = evaluate(&parse_capture(&input).unwrap()).unwrap();
        assert!(report.route.starts_with("dense_"));
        assert_eq!(report.final_keys, 3);
        assert_eq!(report.measured_dense_difference, 0.0);
    }

    #[test]
    fn malformed_or_forged_head_mapping_and_extra_payload_rejected() {
        assert!(parse_capture(&fixture("smollm2-135m-instruct", 9, 3, 4, 2)).is_err());
        assert!(parse_capture(&fixture("gpt2-xl", 9, 3, 4, 1)).is_err());
        assert!(parse_capture(&format!("{} 12", fixture("gpt2-xl", 4, 4, 2, 2))).is_err());
        assert!(
            parse_capture(&fixture("gpt2-xl", 4, 4, 2, 2).replace("8 -8 0", "nan -8 0")).is_err()
        );
    }
}
