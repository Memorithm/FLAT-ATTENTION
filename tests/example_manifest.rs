//! Inventory gate: every `examples/*.rs` stem must have a Cargo.toml
//! `[[example]]` entry so required-features cannot drift silently.

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

fn declared_example_names(toml: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut in_example = false;
    for line in toml.lines() {
        let line = line.trim();
        if line == "[[example]]" {
            in_example = true;
            continue;
        }
        if in_example && line.starts_with("name =") {
            if let Some(name) = line.split('"').nth(1) {
                names.push(name.to_owned());
            }
            in_example = false;
        }
    }
    names.sort();
    names
}

#[test]
fn every_example_source_has_a_cargo_manifest_entry() {
    let toml = fs::read_to_string(root().join("Cargo.toml")).expect("Cargo.toml");
    let declared = declared_example_names(&toml);
    let missing: Vec<_> = example_stems()
        .into_iter()
        .filter(|stem| !declared.iter().any(|name| name == stem))
        .collect();
    assert!(
        missing.is_empty(),
        "examples/*.rs missing from Cargo.toml [[example]]: {missing:?}"
    );
}
