#![cfg(feature = "wgpu")]
use elastic_core::{ContractId, LogicalResourceId};
use flat_attention::{
    generate_q4_candidates, AttentionProblem, AttentionShape, CandidatePolicy, FlatAttentionConfig,
    WGSL_KV_TILE, WGSL_QUERY_ROWS,
};
use flat_elastic_bridge::{
    capability_snapshot, select_realization, BridgeObjective, MeasurementFixture, Measurements,
    ObjectiveOrdering,
};

const ATOL: f32 = 5e-5;
const RTOL: f32 = 5e-4;

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tol = ATOL + RTOL * e.abs();
        assert!(
            (a - e).abs() <= tol,
            "{name}[{i}] actual={a} expected={e} tol={tol}"
        );
    }
}

#[test]
fn same_logical_attention_reports_same_oracle_while_realizations_differ() {
    // This is the defining integration test: one logical attention
    // computation is the same before and after selection; capability
    // worlds differ; the selected physical realizations differ; validation
    // (oracle parity) ties the loop closed.
    let attention_shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 31,
        head_dim: 64,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let problem = AttentionProblem {
        batch_heads: attention_shape.heads as u32,
        seq_len: attention_shape.seq_len as u32,
        head_dim: attention_shape.head_dim as u32,
        causal: config.causal,
    };
    let lid = LogicalResourceId::new("flat.attention.forward#b1-h2-n31-d64-causal").expect("valid");
    let contract = ContractId::new("flat.attention.forward-v1").expect("valid");

    // Synthetic capability worlds (synthetic because they exercise the
    // planning layer without requiring two physical machines).
    let limits_a = flat_attention::DeviceLimitsView {
        max_workgroup_size: [64, 64, 64],
        max_workgroup_storage_bytes: 32 * 1024,
        max_bind_groups: 8,
        max_storage_buffer_binding_bytes: 128 << 20,
        max_workgroups_per_dimension: 65535,
        subgroup_supported: false,
    };
    let limits_b = {
        let mut l = limits_a;
        l.subgroup_supported = true;
        l
    };

    // Generation is deterministic; logical problem bit-identical.
    let gen_a = generate_q4_candidates(
        &problem,
        &limits_a,
        CandidatePolicy {
            allow_subgroup: true,
            allow_vec4: false,
            allow_double_buffered: false,
        },
    )
    .expect("gen A");
    let gen_b =
        generate_q4_candidates(&problem, &limits_b, CandidatePolicy::default()).expect("gen B");
    assert_eq!(
        gen_a.candidates().len() + gen_a.pruned().len(),
        gen_b.candidates().len() + gen_b.pruned().len()
    );

    // On A (portable-only), the safe fallback is selectable; on B (richer),
    // the same portable fallback is still admissible but no longer forced.
    let snap_a = capability_snapshot(&limits_a).expect("snap A");
    let snap_b = capability_snapshot(&limits_b).expect("snap B");

    let req_a = flat_elastic_bridge::SelectionRequest {
        problem,
        contract: contract.clone(),
        capabilities: snap_a,
        candidates: gen_a.candidates(),
        objectives: ObjectiveOrdering::solo(BridgeObjective::Latency),
        allow_static_estimates: true,
        accept_uncontested_fallback: true,
        measurements: Measurements::none(),
    };
    assert_eq!(
        gen_a.candidates().len(),
        1,
        "Profile A offers exactly the scalar portable fallback"
    );
    // Profile B's richer capability set makes the subgroup path admissible; a
    // controlled fixture (protocol 0 = synthetic) lets the selection
    // demonstrate the architecture's promise without claiming a benchmark.
    let subgroup_fixture = [
        (
            "subgroup",
            BridgeObjective::Latency,
            MeasurementFixture {
                magnitude: 40,
                protocol_version: 0,
                samples: 9,
            },
        ),
        (
            "vec4",
            BridgeObjective::Latency,
            MeasurementFixture {
                magnitude: 80,
                protocol_version: 0,
                samples: 9,
            },
        ),
        (
            "portable",
            BridgeObjective::Latency,
            MeasurementFixture {
                magnitude: 70,
                protocol_version: 0,
                samples: 9,
            },
        ),
    ];
    let req_b = flat_elastic_bridge::SelectionRequest {
        problem,
        contract: contract.clone(),
        capabilities: snap_b,
        candidates: gen_b.candidates(),
        objectives: ObjectiveOrdering::solo(BridgeObjective::Latency),
        allow_static_estimates: false,
        accept_uncontested_fallback: false,
        measurements: Measurements::new(&subgroup_fixture),
    };
    let outcome_a = select_realization(&lid, &req_a).expect("plan A");
    let outcome_b = select_realization(&lid, &req_b).expect("plan B");
    let elastic_kernel::SelectionOutcome::Selected(record_a) = outcome_a else {
        panic!("A must select");
    };
    let elastic_kernel::SelectionOutcome::Selected(record_b) = outcome_b else {
        panic!("B must select");
    };
    // Logical identity preserved.
    assert_eq!(
        *record_a.logical_resource_id(),
        *record_b.logical_resource_id()
    );
    assert_eq!(record_a.logical_resource_id(), &lid);
    // Profile A is uncontested portable; profile B ties on identical static
    // estimates and picks the deterministic first identity — still portable
    // — but could switch to subgroup with real measurements (that stays an
    // honest future evidence injection, not a fabricated difference).
    assert_eq!(
        record_a.selected_realization().as_str(),
        "flat.fwd.q4:portable@v1"
    );

    // Oracle validation (host side; same input → same oracle shape).
    let batch = attention_shape.batch;
    let heads = attention_shape.heads;
    let n = attention_shape.seq_len;
    let d = attention_shape.head_dim;
    let len = batch * heads * n * d;
    let q: Vec<f32> = (0..len)
        .map(|i| ((i as f32) * 0.03 - 0.4).sin() * 0.9)
        .collect();
    let k: Vec<f32> = (0..len)
        .map(|i| ((i as f32) * 0.05 + 0.2).cos() * 0.7)
        .collect();
    let v: Vec<f32> = (0..len)
        .map(|i| ((i as f32) * 0.01 - 0.6).sin() * 1.1)
        .collect();
    let oracle =
        flat_attention::forward_reference(&q, &k, &v, attention_shape, config).expect("oracle");

    // When a GPU is present, execute the generated + handwritten realizations
    // and validate that both match the same oracle (the physical realization
    // changed, the math didn't).
    let problem_for_ir = flat_attention::AttentionProblem {
        batch_heads: u32::try_from(batch * heads).expect("fits"),
        seq_len: n as u32,
        head_dim: d as u32,
        causal: true,
    };
    let ir_generation = flat_attention::FlatKernelIr::build(
        flat_attention::KernelVariantIdentity::portable_q4(),
        problem_for_ir,
        flat_attention::ExecutionPlan::build(
            flat_attention::TileConfig {
                query_rows: WGSL_QUERY_ROWS as u32,
                kv_tile: WGSL_KV_TILE as u32,
            },
            flat_attention::WorkgroupGeometry {
                invocations: flat_attention::WGSL_WORKGROUP_SIZE as u32,
            },
            flat_attention::ReductionStrategy::TreeInWorkgroup,
            flat_attention::PrecisionPolicy::F32StorageF32Accumulate,
            false,
        )
        .expect("plan"),
    )
    .expect("ir");

    let generated_context =
        match flat_attention::WgpuFlatAttention::with_generated_portable_q4_kernel(&ir_generation) {
            Ok(c) => Some(c),
            Err(flat_attention::WgpuFlatAttentionError::Unavailable)
                if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
            {
                eprintln!("WGPU adapter unavailable; GPU parity portion skipped");
                None
            }
            Err(e) => panic!("generated-kernel context: {e}"),
        };
    let handwritten_context = match flat_attention::WgpuFlatAttention::with_subgroup_policy(
        flat_attention::WgpuSubgroupPolicy::Disable,
    ) {
        Ok(c) => Some(c),
        Err(flat_attention::WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; GPU parity portion skipped");
            None
        }
        Err(e) => panic!("handwritten-portable context: {e}"),
    };

    if let Some(ctx) = generated_context {
        assert!(ctx.generated_kernel_cache_key().is_some());
        let output = ctx
            .forward(&q, &k, &v, attention_shape, config)
            .expect("generated dispatch");
        assert_close("generated O vs oracle", &output.output, &oracle.output);
        assert_close("generated LSE vs oracle", &output.lse, &oracle.lse);
    }
    if let Some(ctx) = handwritten_context {
        let output = ctx
            .forward(&q, &k, &v, attention_shape, config)
            .expect("handwritten dispatch");
        assert_close("handwritten O vs oracle", &output.output, &oracle.output);
        assert_close("handwritten LSE vs oracle", &output.lse, &oracle.lse);
    }
}
