use super::{import_and_verify, narrow_tensor, FlatParityConfig, GraduationImportError};
use ada_a10_evidence_schema::{
    EvidenceWorkloadFingerprint, SemanticEvidenceRecord, SemanticEvidenceSpec,
};
use ada_cegis::{CegisConfig, CegisEngine};
use ada_core::{
    DiagnosticEvidenceKind, ImplementationCandidateId, QualificationVerdict, SemanticId,
};
use ada_cost_model::{CostAssumptions, OperationProfile};
use ada_graduation::{FlatGraduationBundle, GraduationObjectives};
use ada_implementation::{
    AlgorithmPlan, Buffering, ExpStrategy, ImplementationPlan, MemoryLevel, MemoryPlan,
    ReductionTopology, SchedulePlan, TileShape, WorkPartition,
};
use ada_objective::{MeasuredCost, ObjectiveDirection, QualityMetric};
use ada_qualification::{
    BoundedOracleQualification, EvidenceBoundQualification, NoAdversarialGenerator,
    SemanticWorkloadCase, SemanticWorkloadOracle,
};
use ada_replay::{ReplayCaseSpec, ReplayReferenceInput};
use ada_search::{
    SearchBudget, SearchEngine, SemanticSearchConfig, SemanticSearchSpace, MAX_PROGRAM_COST,
};
use ada_semantic::{InputTransform, MaskRule, SelectionRule, SemanticProgram, WeightRule};
use ada_workload::{
    AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, MaskKind, MaskSpec,
    PrecisionPolicy, ScalarPrecision, SequenceLengths, WorkloadContract, WorkloadOptions,
};

fn workload(topology: AttentionTopology) -> WorkloadContract {
    let geometry = AttentionGeometry::new(GeometrySpec {
        sequence_lengths: SequenceLengths::uniform(1, 2, 2).unwrap(),
        query_heads: 1,
        kv_heads: 1,
        qk_dimension: Some(1),
        value_dimension: 1,
        topology,
        head_grouping: HeadGrouping::MultiHead,
    })
    .unwrap();
    WorkloadContract::new(
        geometry,
        WorkloadOptions {
            mask: MaskSpec::new(MaskKind::None).unwrap(),
            precision: PrecisionPolicy::new(
                ScalarPrecision::F64,
                ScalarPrecision::F64,
                ScalarPrecision::F64,
                ScalarPrecision::F64,
            ),
            ..WorkloadOptions::default()
        },
    )
    .unwrap()
}

fn replay_input(queries: Vec<f64>, keys: Vec<f64>, values: Vec<f64>) -> ReplayReferenceInput {
    ReplayReferenceInput::new(ada_semantic::ReferenceInputSpec {
        query_count: 2,
        key_count: 2,
        q_dimension: 1,
        value_dimension: 1,
        queries,
        keys,
        values,
        external_mask: None,
    })
    .unwrap()
}

fn expected_softmax(queries: &[f64], keys: &[f64], values: &[f64], scale: f64) -> Vec<f64> {
    queries
        .iter()
        .map(|&query| {
            let scores = keys
                .iter()
                .map(|&key| query * key * scale)
                .collect::<Vec<_>>();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores
                .iter()
                .map(|&score| (score - max).exp())
                .collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            weights
                .iter()
                .zip(values)
                .map(|(&weight, &value)| weight * value)
                .sum::<f64>()
                / sum
        })
        .collect()
}

fn search(scale: f64, selection: SelectionRule) -> SearchEngine<SemanticSearchSpace> {
    let space = SemanticSearchSpace::new(SemanticSearchConfig {
        seed: 41,
        input_transforms: vec![InputTransform::Identity],
        affinity_scales: vec![scale],
        masks: vec![MaskRule::Unmasked],
        selections: vec![selection],
        weights: vec![WeightRule::Softmax],
    })
    .unwrap();
    SearchEngine::new(space, SearchBudget::new(4, 4, MAX_PROGRAM_COST).unwrap()).unwrap()
}

fn evidence(
    semantic: SemanticId,
    workload_fingerprint: EvidenceWorkloadFingerprint,
) -> SemanticEvidenceRecord {
    SemanticEvidenceRecord::new(SemanticEvidenceSpec {
        semantic,
        workload: workload_fingerprint,
        kind: DiagnosticEvidenceKind::TaskBehavior,
        producer_repository: "Memorithm/FLAT-ATTENTION".into(),
        producer_revision: "c".repeat(40),
        artifact_identity: "flat-ada-graduation-test-v1".into(),
        intervention_identity: None,
        observation_horizon: None,
        metric_identity: "scalar-parity-fixture-v1".into(),
        sha256_evidence: "d".repeat(64),
        metrics: vec![("fixture_pass".into(), 1.0)],
    })
    .unwrap()
}

fn qualified(
    result: &ada_cegis::CegisResult<SemanticProgram, SemanticWorkloadCase>,
    workload: &WorkloadContract,
) -> EvidenceBoundQualification {
    let survivor = &result.survivors()[0];
    let oracle =
        BoundedOracleQualification::from_cegis_result(result, survivor.fingerprint(), workload)
            .unwrap();
    let record = evidence(
        survivor.candidate().descriptor().id().clone(),
        oracle.workload_fingerprint(),
    );
    oracle.attach_evidence(vec![record]).unwrap()
}

fn implementation(semantic: SemanticId) -> ImplementationPlan {
    ImplementationPlan::new(
        ImplementationCandidateId::new(semantic, "flat-scalar-reference", 1).unwrap(),
        AlgorithmPlan::DenseBlocked,
        SchedulePlan {
            tile: TileShape {
                queries: 2,
                keys: 2,
                values: 1,
            },
            partition: WorkPartition::Serial,
            reduction: ReductionTopology::Serial,
            exp_strategy: ExpStrategy::Standard,
            pipeline_stages: 1,
            vector_width: 1,
            buffering: Buffering::Single,
        },
        MemoryPlan {
            query: MemoryLevel::Global,
            key: MemoryLevel::Global,
            value: MemoryLevel::Global,
            output: MemoryLevel::Global,
            accumulator: MemoryLevel::Register,
            workspace_bytes: 0,
            alignment_bytes: 8,
            kv_page_rows: None,
        },
    )
    .unwrap()
}

fn graduation(
    topology: AttentionTopology,
    selection: SelectionRule,
    scale: f64,
    queries: Vec<f64>,
    keys: Vec<f64>,
    values: Vec<f64>,
    expected: Vec<f64>,
    ada_tolerance: f64,
) -> FlatGraduationBundle {
    let workload = workload(topology);
    let fixture = ReplayCaseSpec {
        workload: workload.clone(),
        input: replay_input(queries, keys, values),
        expected_output: expected,
        max_abs_tolerance: ada_tolerance,
    }
    .into_fixture("flat-e4b-control")
    .unwrap();
    let result = CegisEngine::new(
        search(scale, selection),
        SemanticWorkloadOracle,
        NoAdversarialGenerator,
        CegisConfig::default(),
        vec![fixture],
    )
    .unwrap()
    .run_to_end()
    .unwrap();
    let qualification = qualified(&result, &workload);
    let semantic = qualification
        .oracle()
        .candidate()
        .candidate()
        .descriptor()
        .id()
        .clone();
    FlatGraduationBundle::new(
        &qualification,
        &result,
        implementation(semantic),
        OperationProfile::scaled_dot_softmax(1).unwrap(),
        CostAssumptions::default(),
        GraduationObjectives {
            measured: MeasuredCost::default(),
            quality: vec![QualityMetric::new(
                "fixture_pass",
                Some(1.0),
                ObjectiveDirection::Maximize,
            )
            .unwrap()],
        },
        QualificationVerdict::ContinueResearch,
    )
    .unwrap()
}

#[test]
fn standard_softmax_self_attention_imports_after_exact_ada_replay() {
    let bundle = graduation(
        AttentionTopology::SelfAttention,
        SelectionRule::All,
        1.0,
        vec![0.0, 0.0],
        vec![1.0, -1.0],
        vec![2.0, 4.0],
        vec![3.0, 3.0],
        0.0,
    );
    let imported = import_and_verify(
        &bundle.to_canonical_text(),
        FlatParityConfig::new(0.0).unwrap(),
    )
    .unwrap();
    assert_eq!(imported.report().fixture_count(), 1);
    assert_eq!(imported.report().ada_worst_max_abs_error().to_bits(), 0);
    assert_eq!(
        imported.report().flat_worst_max_abs_difference().to_bits(),
        0
    );
    assert_eq!(
        imported.bundle().to_canonical_text(),
        bundle.to_canonical_text()
    );
}

#[test]
fn non_all_selection_is_valid_ada_but_rejected_by_current_flat_bridge() {
    let bundle = graduation(
        AttentionTopology::SelfAttention,
        SelectionRule::Window { radius: 0 },
        1.0,
        vec![0.0, 0.0],
        vec![1.0, -1.0],
        vec![2.0, 4.0],
        vec![2.0, 4.0],
        0.0,
    );
    assert!(matches!(
        import_and_verify(
            &bundle.to_canonical_text(),
            FlatParityConfig::new(0.0).unwrap()
        ),
        Err(GraduationImportError::UnsupportedSemantic(_))
    ));
}

#[test]
fn cross_attention_is_valid_ada_but_rejected_by_current_flat_bridge() {
    let bundle = graduation(
        AttentionTopology::CrossAttention,
        SelectionRule::All,
        1.0,
        vec![0.0, 0.0],
        vec![1.0, -1.0],
        vec![2.0, 4.0],
        vec![3.0, 3.0],
        0.0,
    );
    assert!(matches!(
        import_and_verify(
            &bundle.to_canonical_text(),
            FlatParityConfig::new(0.0).unwrap()
        ),
        Err(GraduationImportError::UnsupportedWorkload(_))
    ));
}

#[test]
fn f32_parity_tolerance_is_separate_from_ada_oracle_tolerance() {
    let queries = vec![0.1, -0.2];
    let keys = vec![0.3, -0.4];
    let values = vec![0.7, -1.1];
    let scale = 0.3;
    let expected = expected_softmax(&queries, &keys, &values, scale);
    let bundle = graduation(
        AttentionTopology::SelfAttention,
        SelectionRule::All,
        scale,
        queries,
        keys,
        values,
        expected,
        1.0e-14,
    );

    assert!(matches!(
        import_and_verify(
            &bundle.to_canonical_text(),
            FlatParityConfig::new(0.0).unwrap()
        ),
        Err(GraduationImportError::FlatParityMismatch { .. })
    ));
    let imported = import_and_verify(
        &bundle.to_canonical_text(),
        FlatParityConfig::new(1.0e-5).unwrap(),
    )
    .unwrap();
    assert!(imported.report().ada_worst_max_abs_error() <= 1.0e-14);
    assert!(imported.report().flat_worst_max_abs_difference() > 0.0);
    assert!(imported.report().flat_worst_max_abs_difference() <= 1.0e-5);
}

#[test]
fn malformed_bundle_and_invalid_parity_config_fail_closed() {
    assert_eq!(
        FlatParityConfig::new(f64::NAN).unwrap_err(),
        GraduationImportError::InvalidParityTolerance
    );
    assert!(matches!(
        import_and_verify(
            "not-an-ada-graduation\n",
            FlatParityConfig::new(1.0e-5).unwrap()
        ),
        Err(GraduationImportError::InvalidGraduation(_))
    ));
}

#[test]
fn narrowing_finite_f64_overflow_fails_closed() {
    assert!(matches!(
        narrow_tensor("Q", &[f64::MAX]),
        Err(GraduationImportError::F32NarrowingOverflow {
            field: "Q",
            index: 0,
            ..
        })
    ));
}
