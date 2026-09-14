//! Inventory gate: every root `examples/*.rs` stem must appear in
//! `examples/README.md`. Presence in that table is documentation, not routing.

use std::fs;
use std::path::PathBuf;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

fn examples_readme() -> String {
    let path = examples_dir().join("README.md");
    fs::read_to_string(path).expect("examples/README.md must exist")
}

fn tracked_example_stems() -> Vec<String> {
    let mut names = Vec::new();
    for entry in fs::read_dir(examples_dir()).expect("examples/ must exist") {
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

fn backticked_example_stems(readme: &str) -> Vec<&str> {
    let mut stems = Vec::new();
    let mut rest = readme;
    while let Some(start) = rest.find('`') {
        rest = &rest[start + 1..];
        let Some(end) = rest.find('`') else {
            break;
        };
        let inner = &rest[..end];
        rest = &rest[end + 1..];
        if is_example_stem(inner) {
            stems.push(inner);
        }
    }
    stems
}

fn is_example_stem(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    name.chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        && name.contains('_')
}

#[test]
fn every_root_example_is_named_in_examples_readme() {
    let readme = examples_readme();
    let missing: Vec<_> = tracked_example_stems()
        .into_iter()
        .filter(|name| !readme.contains(name.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "examples/*.rs stems missing from examples/README.md: {missing:?}"
    );
}

#[test]
fn examples_readme_does_not_name_absent_example_stems() {
    let readme = examples_readme();
    let present = tracked_example_stems();
    let mut stale = Vec::new();
    for name in backticked_example_stems(&readme) {
        if !present.iter().any(|item| item == name) {
            stale.push(name.to_owned());
        }
    }
    stale.sort();
    stale.dedup();
    assert!(
        stale.is_empty(),
        "examples/README.md names example stems that are not in examples/: {stale:?}"
    );
}
