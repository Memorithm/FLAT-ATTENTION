//! Inventory gate: every `examples/*.rs` stem must have a Cargo.toml
//! `[[example]]` entry so required-features cannot drift silently.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn example_stems() -> Vec<String> {
    let mut names = Vec::new();
    let dir = root().join("examples");
    for entry in fs::read_dir(dir).expect("examples/ must exist") {
        let entry = entry.expect("readable examples/ entry");
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(stem) = name.strip_suffix(".rs") {
            names.push(stem.to_owned());
        }
    }
    names.sort();
    names
}

#[derive(Debug, Clone)]
struct ExampleManifest {
    name: String,
    requires_wgpu: bool,
}

fn declared_examples(toml: &str) -> Vec<ExampleManifest> {
    let mut examples = Vec::new();
    let mut current: Option<ExampleManifest> = None;
    for line in toml.lines() {
        let line = line.trim();
        if line == "[[example]]" {
            if let Some(example) = current.take() {
                examples.push(example);
            }
            current = Some(ExampleManifest {
                name: String::new(),
                requires_wgpu: false,
            });
            continue;
        }
        let Some(example) = current.as_mut() else {
            continue;
        };
        if line.starts_with("name =") {
            if let Some(name) = line.split('"').nth(1) {
                example.name = name.to_owned();
            }
        }
        if line.contains("required-features") && line.contains("wgpu") {
            example.requires_wgpu = true;
        }
    }
    if let Some(example) = current {
        examples.push(example);
    }
    examples
}

fn readme_section_stems(readme: &str, heading: &str) -> Vec<String> {
    let start = readme.find(heading).expect("examples README heading");
    let rest = &readme[start + heading.len()..];
    let end = rest.find("\n## ").unwrap_or(rest.len());
    let section = &rest[..end];
    let mut stems = Vec::new();
    let mut cursor = section;
    while let Some(open) = cursor.find('`') {
        cursor = &cursor[open + 1..];
        let Some(close) = cursor.find('`') else {
            break;
        };
        let inner = &cursor[..close];
        cursor = &cursor[close + 1..];
        if inner
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
            && inner.contains('_')
        {
            stems.push(inner.to_owned());
        }
    }
    stems.sort();
    stems.dedup();
    stems
}

#[test]
fn every_example_source_has_a_cargo_manifest_entry() {
    let toml = fs::read_to_string(root().join("Cargo.toml")).expect("Cargo.toml");
    let declared: Vec<_> = declared_examples(&toml)
        .into_iter()
        .map(|example| example.name)
        .collect();
    let missing: Vec<_> = example_stems()
        .into_iter()
        .filter(|stem| !declared.iter().any(|name| name == stem))
        .collect();
    assert!(
        missing.is_empty(),
        "examples/*.rs missing from Cargo.toml [[example]]: {missing:?}"
    );
}

#[test]
fn readme_host_gpu_split_matches_required_features() {
    let toml = fs::read_to_string(root().join("Cargo.toml")).expect("Cargo.toml");
    let readme = fs::read_to_string(root().join("examples/README.md")).expect("README");
    let by_name: BTreeMap<_, _> = declared_examples(&toml)
        .into_iter()
        .map(|example| (example.name.clone(), example))
        .collect();

    let host = readme_section_stems(&readme, "## Host-only");
    let gpu = readme_section_stems(&readme, "## GPU");
    assert!(!host.is_empty(), "host-only table must name examples");
    assert!(!gpu.is_empty(), "GPU table must name examples");

    let mut host_with_wgpu = Vec::new();
    for name in &host {
        let Some(example) = by_name.get(name) else {
            panic!("host-only README stem missing from Cargo.toml: {name}");
        };
        if example.requires_wgpu {
            host_with_wgpu.push(name.clone());
        }
    }
    assert!(
        host_with_wgpu.is_empty(),
        "host-only examples must not require wgpu: {host_with_wgpu:?}"
    );

    let mut gpu_without_wgpu = Vec::new();
    for name in &gpu {
        let Some(example) = by_name.get(name) else {
            panic!("GPU README stem missing from Cargo.toml: {name}");
        };
        if !example.requires_wgpu {
            gpu_without_wgpu.push(name.clone());
        }
    }
    assert!(
        gpu_without_wgpu.is_empty(),
        "GPU examples must declare required-features wgpu: {gpu_without_wgpu:?}"
    );
}
