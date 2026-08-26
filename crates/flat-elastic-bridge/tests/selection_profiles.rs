#![forbid(unsafe_code)]
use elastic_core::{ContractId, LogicalResourceId};
use flat_attention::{generate_q4_candidates, AttentionProblem, CandidatePolicy, DeviceLimitsView};
use flat_elastic_bridge::{
    capability_snapshot, select_realization, BridgeObjective, MeasurementFixture, Measurements,
    ObjectiveOrdering,
};
const CONTRACT_TEXT: &str = "flat.attention.forward-v1";
fn contract() -> ContractId {
    ContractId::new(CONTRACT_TEXT).expect("valid")
}
fn logical_resource() -> LogicalResourceId {
    LogicalResourceId::new("flat.attention.forward#b1-h8-n128-d64-causal").expect("valid")
}
fn problem() -> AttentionProblem {
    AttentionProblem {
        batch_heads: 8,
        seq_len: 128,
        head_dim: 64,
        causal: true,
    }
}
fn profile_a() -> DeviceLimitsView {
    DeviceLimitsView {
        max_workgroup_size: [256, 256, 64],
        max_workgroup_storage_bytes: 32 * 1024,
        max_bind_groups: 8,
        max_storage_buffer_binding_bytes: 128 << 20,
        max_workgroups_per_dimension: 65535,
        subgroup_supported: false,
    }
}
fn profile_b() -> DeviceLimitsView {
    let mut l = profile_a();
    l.subgroup_supported = true;
    l
}

#[test]
fn profile_a_selects_the_portable_fallback_and_states_it() {
    let policy = CandidatePolicy {
        allow_subgroup: true,
        allow_vec4: false,
        allow_double_buffered: false,
    };
    let report = generate_q4_candidates(&problem(), &profile_a(), policy).expect("generation");
    assert_eq!(
        report.candidates().len(),
        1,
        "only the scalar portable fallback is offered"
    );
    let snapshot = capability_snapshot(&profile_a()).expect("snapshot");
    let outcome = select_realization(
        &logical_resource(),
        &flat_elastic_bridge::SelectionRequest {
            problem: problem(),
            contract: contract(),
            capabilities: snapshot,
            candidates: report.candidates(),
            objectives: ObjectiveOrdering::solo(BridgeObjective::Latency),
            allow_static_estimates: true,
            accept_uncontested_fallback: true,
            measurements: Measurements::none(),
        },
    )
    .expect("selection runs");
    let elastic_kernel::SelectionOutcome::Selected(record) = outcome else {
        panic!("profile A must select its uncontested fallback, got {outcome:?}");
    };
    assert_eq!(
        record.selected_realization().as_str(),
        "flat.fwd.q4:portable@v1"
    );
    assert_eq!(
        record.decisive_evidence(),
        Some(&elastic_kernel::DecisiveEvidence::UncontestedFallback)
    );
    assert!(report
        .pruned()
        .iter()
        .any(|(f, r)| *f == "flat.fwd.q4:subgroup"
            && *r == flat_attention::PrunedReason::SubgroupUnavailable));
}

#[test]
fn profile_b_without_measurements_ties_break_deterministically() {
    let report = generate_q4_candidates(&problem(), &profile_b(), CandidatePolicy::default())
        .expect("generation");
    let snapshot = capability_snapshot(&profile_b()).expect("snapshot");
    let outcome = select_realization(
        &logical_resource(),
        &flat_elastic_bridge::SelectionRequest {
            problem: problem(),
            contract: contract(),
            capabilities: snapshot,
            candidates: report.candidates(),
            objectives: ObjectiveOrdering::solo(BridgeObjective::MemoryFootprint),
            allow_static_estimates: true,
            accept_uncontested_fallback: false,
            measurements: Measurements::none(),
        },
    )
    .expect("selection runs");
    let elastic_kernel::SelectionOutcome::Selected(record) = &outcome else {
        panic!("tie must resolve, got {outcome:?}");
    };
    let selected = record.selected_realization().as_str().to_string();
    assert_eq!(
        selected, "flat.fwd.q4:portable@v1",
        "lexicographic tie-break"
    );
    let replay = select_realization(
        &logical_resource(),
        &flat_elastic_bridge::SelectionRequest {
            problem: problem(),
            contract: contract(),
            capabilities: snapshot,
            candidates: report.candidates(),
            objectives: ObjectiveOrdering::solo(BridgeObjective::MemoryFootprint),
            allow_static_estimates: true,
            accept_uncontested_fallback: false,
            measurements: Measurements::none(),
        },
    )
    .expect("replay");
    assert_eq!(outcome, replay);
}

#[test]
fn measured_fixture_evidence_selects_the_subgroup_realization_on_profile_b() {
    let report = generate_q4_candidates(&problem(), &profile_b(), CandidatePolicy::default())
        .expect("generation");
    let snapshot = capability_snapshot(&profile_b()).expect("snapshot");
    let fixture_protocol = 0u32;
    let entries = [
        (
            "subgroup",
            BridgeObjective::Latency,
            MeasurementFixture {
                magnitude: 40,
                protocol_version: fixture_protocol,
                samples: 9,
            },
        ),
        (
            "vec4",
            BridgeObjective::Latency,
            MeasurementFixture {
                magnitude: 90,
                protocol_version: fixture_protocol,
                samples: 9,
            },
        ),
        (
            "portable",
            BridgeObjective::Latency,
            MeasurementFixture {
                magnitude: 100,
                protocol_version: fixture_protocol,
                samples: 9,
            },
        ),
    ];
    let outcome = select_realization(
        &logical_resource(),
        &flat_elastic_bridge::SelectionRequest {
            problem: problem(),
            contract: contract(),
            capabilities: snapshot,
            candidates: report.candidates(),
            objectives: ObjectiveOrdering::solo(BridgeObjective::Latency),
            allow_static_estimates: false,
            accept_uncontested_fallback: false,
            measurements: Measurements::new(&entries),
        },
    )
    .expect("selection runs");
    let elastic_kernel::SelectionOutcome::Selected(record) = outcome else {
        panic!("profile B must select, got {outcome:?}");
    };
    assert_eq!(
        record.selected_realization().as_str(),
        "flat.fwd.q4:subgroup@v1"
    );
    assert!(matches!(
        record.decisive_evidence(),
        Some(elastic_kernel::DecisiveEvidence::Measured { .. })
    ));
}

#[test]
fn objective_priority_flips_the_selected_realization_legally() {
    let report = generate_q4_candidates(&problem(), &profile_b(), CandidatePolicy::default())
        .expect("generation");
    let snapshot = capability_snapshot(&profile_b()).expect("snapshot");
    let entries = [
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
        (
            "subgroup",
            BridgeObjective::MemoryFootprint,
            MeasurementFixture {
                magnitude: 9000,
                protocol_version: 0,
                samples: 9,
            },
        ),
        (
            "vec4",
            BridgeObjective::MemoryFootprint,
            MeasurementFixture {
                magnitude: 3000,
                protocol_version: 0,
                samples: 9,
            },
        ),
        (
            "portable",
            BridgeObjective::MemoryFootprint,
            MeasurementFixture {
                magnitude: 6000,
                protocol_version: 0,
                samples: 9,
            },
        ),
    ];
    let latency_first = select_realization(
        &logical_resource(),
        &flat_elastic_bridge::SelectionRequest {
            problem: problem(),
            contract: contract(),
            capabilities: snapshot,
            candidates: report.candidates(),
            objectives: ObjectiveOrdering::pair(
                BridgeObjective::Latency,
                BridgeObjective::MemoryFootprint,
            ),
            allow_static_estimates: false,
            accept_uncontested_fallback: false,
            measurements: Measurements::new(&entries),
        },
    )
    .expect("selection runs");
    let elastic_kernel::SelectionOutcome::Selected(r) = latency_first else {
        panic!("latency-first must select");
    };
    assert_eq!(r.selected_realization().as_str(), "flat.fwd.q4:subgroup@v1");
    let memory_first = select_realization(
        &logical_resource(),
        &flat_elastic_bridge::SelectionRequest {
            problem: problem(),
            contract: contract(),
            capabilities: snapshot,
            candidates: report.candidates(),
            objectives: ObjectiveOrdering::pair(
                BridgeObjective::MemoryFootprint,
                BridgeObjective::Latency,
            ),
            allow_static_estimates: false,
            accept_uncontested_fallback: false,
            measurements: Measurements::new(&entries),
        },
    )
    .expect("selection runs");
    let elastic_kernel::SelectionOutcome::Selected(r) = memory_first else {
        panic!("memory-first must select");
    };
    assert_eq!(r.selected_realization().as_str(), "flat.fwd.q4:vec4@v1");
}

#[test]
fn logical_identity_is_preserved_while_realizations_change() {
    let wf_a = flat_elastic_bridge::workload_fingerprint(&problem(), &contract());
    let wf_b = flat_elastic_bridge::workload_fingerprint(&problem(), &contract());
    assert_eq!(wf_a, wf_b);
    let on_profile_a = {
        let policy = CandidatePolicy {
            allow_subgroup: true,
            allow_vec4: false,
            allow_double_buffered: false,
        };
        let report = generate_q4_candidates(&problem(), &profile_a(), policy).expect("gen");
        let snap = capability_snapshot(&profile_a()).expect("snap");
        let o = select_realization(
            &logical_resource(),
            &flat_elastic_bridge::SelectionRequest {
                problem: problem(),
                contract: contract(),
                capabilities: snap,
                candidates: report.candidates(),
                objectives: ObjectiveOrdering::solo(BridgeObjective::Latency),
                allow_static_estimates: true,
                accept_uncontested_fallback: true,
                measurements: Measurements::none(),
            },
        )
        .expect("o");
        let elastic_kernel::SelectionOutcome::Selected(r) = &o else {
            panic!();
        };
        assert_eq!(*r.logical_resource_id(), logical_resource());
        r.selected_realization().as_str().to_string()
    };
    let on_profile_b = {
        let report = generate_q4_candidates(&problem(), &profile_b(), CandidatePolicy::default())
            .expect("gen");
        let snap = capability_snapshot(&profile_b()).expect("snap");
        let o = select_realization(
            &logical_resource(),
            &flat_elastic_bridge::SelectionRequest {
                problem: problem(),
                contract: contract(),
                capabilities: snap,
                candidates: report.candidates(),
                objectives: ObjectiveOrdering::solo(BridgeObjective::MemoryFootprint),
                allow_static_estimates: true,
                accept_uncontested_fallback: false,
                measurements: Measurements::none(),
            },
        )
        .expect("o");
        let elastic_kernel::SelectionOutcome::Selected(r) = &o else {
            panic!();
        };
        assert_eq!(*r.logical_resource_id(), logical_resource());
        r.selected_realization().as_str().to_string()
    };
    assert_eq!(on_profile_a, "flat.fwd.q4:portable@v1");
    assert_eq!(on_profile_b, "flat.fwd.q4:portable@v1");
    assert_eq!(wf_a.bits(), wf_b.bits());
}
