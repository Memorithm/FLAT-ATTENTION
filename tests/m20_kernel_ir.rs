//! M20/M21/M25 first-slice tests: kernel IR, deterministic candidate
//! generation, and WGSL emission.
//!
//! Everything here is host-only: capability profiles are synthetic
//! [`DeviceLimitsView`] values and shader validation uses Naga directly. These
//! are planning/codegen tests, not hardware-performance claims.

use flat_attention::{
    generate_q4_candidates, AttentionProblem, CandidatePolicy, DeviceLimitsView, FlatKernelIr,
    KernelIrError, KernelVariantIdentity, PrecisionPolicy, ReductionStrategy,
};

fn portable_limits() -> DeviceLimitsView {
    DeviceLimitsView {
        max_workgroup_size: [128, 128, 64],
        max_workgroup_storage_bytes: 32 * 1024,
        max_bind_groups: 8,
        max_storage_buffer_binding_bytes: 128 << 20,
        max_workgroups_per_dimension: 65535,
        subgroup_supported: false,
    }
}

fn subgroup_limits() -> DeviceLimitsView {
    let mut limits = portable_limits();
    limits.subgroup_supported = true;
    limits
}

fn problem(head_dim: u32, seq_len: u32) -> AttentionProblem {
    AttentionProblem {
        batch_heads: 4,
        seq_len,
        head_dim,
        causal: true,
    }
}

#[test]
fn problem_converts_from_public_shape_contract() {
    let shape = flat_attention::AttentionShape {
        batch: 1,
        heads: 8,
        seq_len: 129,
        head_dim: 64,
    };
    let converted =
        AttentionProblem::from_shape(&shape, flat_attention::FlatAttentionConfig::default())
            .expect("converts");
    assert_eq!(converted.batch_heads, 8);
    assert_eq!(converted.seq_len, 129);
}

#[test]
fn ir_rejects_illegal_descriptions_before_generation() {
    let tiles = flat_attention::TileConfig {
        query_rows: 4,
        kv_tile: 8,
    };
    // Subgroup reduction without a subgroup requirement must fail in the IR.
    assert_eq!(
        flat_attention::ExecutionPlan::build(
            tiles,
            flat_attention::WorkgroupGeometry { invocations: 64 },
            ReductionStrategy::SubgroupArithmetic,
            PrecisionPolicy::F32StorageF32Accumulate,
            false,
        ),
        Err(KernelIrError::SubgroupOperationWithoutRequirement)
    );
    // Zero tile dimension.
    assert!(flat_attention::ExecutionPlan::build(
        flat_attention::TileConfig {
            query_rows: 0,
            kv_tile: 8,
        },
        flat_attention::WorkgroupGeometry { invocations: 64 },
        ReductionStrategy::TreeInWorkgroup,
        PrecisionPolicy::F32StorageF32Accumulate,
        false,
    )
    .is_err());
    // Identity schema mismatch.
    let plan = flat_attention::ExecutionPlan::build(
        tiles,
        flat_attention::WorkgroupGeometry { invocations: 64 },
        ReductionStrategy::TreeInWorkgroup,
        PrecisionPolicy::F32StorageF32Accumulate,
        false,
    )
    .expect("plan");
    assert_eq!(
        FlatKernelIr::build(
            KernelVariantIdentity {
                family: "flat.fwd.q4",
                variant: "portable",
                schema_version: 99,
            },
            problem(64, 64),
            plan,
        ),
        Err(KernelIrError::SchemaVersionMismatch {
            actual: 99,
            expected: flat_attention::KERNEL_IR_SCHEMA_VERSION,
        })
    );
}

#[test]
fn identical_configuration_produces_identical_ir_and_fingerprint() {
    let build = || {
        FlatKernelIr::build(
            KernelVariantIdentity::portable_q4(),
            problem(64, 128),
            flat_attention::ExecutionPlan::build(
                flat_attention::TileConfig {
                    query_rows: 4,
                    kv_tile: 8,
                },
                flat_attention::WorkgroupGeometry { invocations: 64 },
                ReductionStrategy::TreeInWorkgroup,
                PrecisionPolicy::F32StorageF32Accumulate,
                false,
            )
            .expect("plan"),
        )
        .expect("ir")
    };
    let first = build();
    let second = build();
    assert_eq!(first, second);
    assert_eq!(first.fingerprint(), second.fingerprint());
}

#[test]
fn candidate_sets_differ_across_capability_profiles() {
    // Profile A (portable): only generated + vec4-eligible families survive;
    // subgroup is pruned with an explicit reason.
    let report_a = generate_q4_candidates(
        &problem(64, 96),
        &portable_limits(),
        CandidatePolicy::default(),
    )
    .expect("generation succeeds");
    let sources_a: Vec<_> = report_a
        .candidates()
        .iter()
        .map(|spec| (spec.ir().identity().variant, spec.source()))
        .collect();
    assert!(!sources_a.is_empty());
    assert!(report_a
        .pruned()
        .iter()
        .any(|(family, reason)| *family == "flat.fwd.q4:subgroup"
            && *reason == flat_attention::PrunedReason::SubgroupUnavailable));

    // Profile B (subgroup capable): the subgroup family becomes admissible.
    let report_b = generate_q4_candidates(
        &problem(64, 96),
        &subgroup_limits(),
        CandidatePolicy::default(),
    )
    .expect("generation succeeds");
    assert!(report_b.candidates().len() > report_a.candidates().len());
    assert!(
        report_b
            .candidates()
            .first()
            .expect("ordered")
            .requirements()
            .requires_subgroup
    );

    // The logical attention problem is bit-identical across both reports:
    // only admissible realizations changed.
    for spec in report_b.candidates() {
        assert_eq!(spec.ir().problem(), &problem(64, 96));
    }

    // Deterministic generation: same inputs, same ordered output.
    let replay = generate_q4_candidates(
        &problem(64, 96),
        &subgroup_limits(),
        CandidatePolicy::default(),
    )
    .expect("replay");
    assert_eq!(report_b, replay);
}

#[test]
fn odd_head_dimensions_prune_the_vec4_family_explicitly() {
    let report = generate_q4_candidates(
        &problem(66, 32),
        &subgroup_limits(),
        CandidatePolicy {
            allow_subgroup: true,
            allow_double_buffered: true,
            allow_vec4: true,
        },
    )
    .expect("generation succeeds");
    assert!(report.pruned().iter().any(|(family, reason)| {
        (*family == "flat.fwd.q4:vec4" || *family == "flat.fwd.q4:double-buffer")
            && matches!(
                reason,
                flat_attention::PrunedReason::HeadDimNotMultiple { actual: 66, .. }
            )
    }));
    // The scalar families remain legal for head_dim=66.
    assert!(report.has_candidates());
}

#[test]
fn tiny_workgroup_storage_prunes_every_family_honestly() {
    let mut limits = subgroup_limits();
    limits.max_workgroup_storage_bytes = 1024;
    let report = generate_q4_candidates(&problem(64, 16), &limits, CandidatePolicy::default())
        .expect("problem-level checks still pass");
    assert!(!report.has_candidates());
    // Every family including the always-attempted portable generated path
    // is storage-pruned with an explicit reason.
    assert_eq!(report.pruned().len(), 4);
    assert!(report.pruned().iter().any(|(family, reason)| {
        *family == "flat.fwd.q4:portable"
            && matches!(
                reason,
                flat_attention::PrunedReason::WorkgroupStorageExceeded {
                    required_bytes: 11_328,
                    available_bytes: 1024,
                }
            )
    }));
}

#[test]
fn generated_wgsl_is_deterministic_and_validated_by_naga() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let ir = FlatKernelIr::build(
        KernelVariantIdentity::portable_q4(),
        problem(64, 33),
        flat_attention::ExecutionPlan::build(
            flat_attention::TileConfig {
                query_rows: 4,
                kv_tile: 8,
            },
            flat_attention::WorkgroupGeometry { invocations: 64 },
            ReductionStrategy::TreeInWorkgroup,
            PrecisionPolicy::F32StorageF32Accumulate,
            false,
        )
        .expect("plan"),
    )
    .expect("ir");

    let first = flat_attention::emit_wgsl(&ir).expect("emits");
    let second = flat_attention::emit_wgsl(&ir).expect("emits");
    assert_eq!(first.source(), second.source());
    assert_eq!(first.source_fingerprint(), second.source_fingerprint());
    assert_eq!(first.cache_key(), second.cache_key());

    let module = naga::front::wgsl::parse_str(first.source())
        .unwrap_or_else(|err| panic!("generated WGSL parse failed: {err:?}"));
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|err| panic!("generated WGSL validation failed: {err:?}"));
}

#[test]
fn emitter_refuses_strategies_outside_its_qualified_subset() {
    let subgroup_plan = flat_attention::ExecutionPlan::build(
        flat_attention::TileConfig {
            query_rows: 4,
            kv_tile: 8,
        },
        flat_attention::WorkgroupGeometry { invocations: 64 },
        ReductionStrategy::SubgroupArithmetic,
        PrecisionPolicy::F32StorageF32Accumulate,
        true,
    )
    .expect("subgroup plan");
    let subgroup_ir = FlatKernelIr::build(
        KernelVariantIdentity::subgroup_q4(),
        problem(64, 64),
        subgroup_plan,
    )
    .expect("ir");
    let error = flat_attention::emit_wgsl(&subgroup_ir).expect_err("must refuse");
    assert!(matches!(
        error,
        flat_attention::EmitError::UnsupportedSubset { .. }
    ));

    // f16 precision is representable but not emitted either.
    let f16_plan = flat_attention::ExecutionPlan::build(
        flat_attention::TileConfig {
            query_rows: 4,
            kv_tile: 8,
        },
        flat_attention::WorkgroupGeometry { invocations: 64 },
        ReductionStrategy::TreeInWorkgroup,
        PrecisionPolicy::PackedF16StorageF32Accumulate,
        false,
    )
    .expect("f16 plan");
    let f16_ir = FlatKernelIr::build(
        KernelVariantIdentity {
            family: "flat.fwd.q4",
            variant: "f16",
            schema_version: flat_attention::KERNEL_IR_SCHEMA_VERSION,
        },
        problem(64, 64),
        f16_plan,
    )
    .expect("f16 ir");
    let error = flat_attention::emit_wgsl(&f16_ir).expect_err("must refuse");
    assert!(matches!(
        error,
        flat_attention::EmitError::UnsupportedSubset { .. }
    ));
}

#[test]
fn cache_key_tracks_ir_and_codegen_version_but_not_problem_only_noise() {
    let base_ir = FlatKernelIr::build(
        KernelVariantIdentity::portable_q4(),
        problem(64, 64),
        flat_attention::ExecutionPlan::build(
            flat_attention::TileConfig {
                query_rows: 4,
                kv_tile: 8,
            },
            flat_attention::WorkgroupGeometry { invocations: 64 },
            ReductionStrategy::TreeInWorkgroup,
            PrecisionPolicy::F32StorageF32Accumulate,
            false,
        )
        .expect("plan"),
    )
    .expect("ir");
    let key = flat_attention::KernelCacheKey::from_ir(&base_ir);
    assert_eq!(
        key.backend_codegen_version(),
        flat_attention::BACKEND_CODEGEN_VERSION
    );
    assert_eq!(key.ir_fingerprint(), base_ir.fingerprint());

    // A different seq_len changes the problem but NOT the generated kernel
    // text (geometry is dynamic via params), so the IR fingerprint changes
    // while the specialization-relevant content stays comparable through the
    // cache-key fields. The key must track the IR fingerprint exactly.
    let other_ir = FlatKernelIr::build(
        KernelVariantIdentity::portable_q4(),
        problem(64, 65),
        flat_attention::ExecutionPlan::build(
            flat_attention::TileConfig {
                query_rows: 4,
                kv_tile: 8,
            },
            flat_attention::WorkgroupGeometry { invocations: 64 },
            ReductionStrategy::TreeInWorkgroup,
            PrecisionPolicy::F32StorageF32Accumulate,
            false,
        )
        .expect("plan"),
    )
    .expect("ir");
    assert_ne!(
        flat_attention::KernelCacheKey::from_ir(&other_ir).bits(),
        key.bits()
    );
}
