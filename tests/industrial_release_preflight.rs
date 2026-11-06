//! Release-readiness boundary: an incomplete candidate must remain a prerelease.
//! This is a policy regression, not proof that external benchmarks or SciAgent
//! integrations have run. Those require exact-SHA evidence independently.

use std::fs;
use std::path::Path;

fn root_file(path: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_else(|error| panic!("cannot read {path}: {error}"))
}

fn unchecked_items(checklist: &str) -> Vec<&str> {
    checklist
        .lines()
        .filter(|line| line.trim_start().starts_with("- [ ] "))
        .collect()
}

#[test]
fn incomplete_release_checklist_cannot_claim_general_availability() {
    let checklist = root_file("docs/RELEASE_CHECKLIST.md");
    let dossier = root_file("docs/INDUSTRIAL_RELEASE_DOSSIER.md");
    let cargo = root_file("Cargo.toml");
    let unresolved = unchecked_items(&checklist);

    assert!(
        !unresolved.is_empty(),
        "if all release items are checked, replace this prerelease policy test with an exact-SHA release qualification"
    );
    assert!(
        cargo.lines().any(|line| line.trim() == "publish = false"),
        "unqualified 1.0 candidate must not be publishable"
    );
    assert!(
        cargo.lines().any(|line| line.trim() == "version = \"0.1.0\""),
        "unexpected version change requires release-policy review"
    );
    assert!(
        dossier.contains("status: BLOCKED")
            && dossier.contains("release_authorized: false"),
        "industrial dossier must keep the unqualified release fail-closed"
    );
    assert!(
        dossier.contains("Memorithm/scirust/issues/1732")
            && dossier.contains("Memorithm/FLAT-ATTENTION/issues/359"),
        "industrial dossier must link the destination and producer qualification gates"
    );
}

#[test]
fn release_gate_covers_integration_benchmark_and_final_head() {
    let checklist = root_file("docs/RELEASE_CHECKLIST.md");
    for guard in [
        "SciAgent decode/KV lifecycle qualification",
        "SciRust stable FLAT adapter",
        "Measured improvement over SciRust",
        "All required checks are green on that SHA",
        "The tag is created only after all mandatory boxes above are complete",
    ] {
        assert!(
            checklist.contains(guard),
            "mandatory release check missing: {guard}"
        );
    }
}

#[test]
fn release_evidence_is_distinct_from_software_adapter_correctness() {
    let dossier = root_file("docs/INDUSTRIAL_RELEASE_DOSSIER.md");
    for guard in [
        "physical-device benchmark",
        "software Vulkan",
        "exact FLAT commit",
        "SciRust lockfile",
        "no runtime promotion",
    ] {
        assert!(dossier.contains(guard), "missing evidence boundary: {guard}");
    }
}
