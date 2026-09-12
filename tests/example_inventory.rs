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

fn markdown_example_token(token: &str) -> Option<&str> {
    let name = token.trim_matches('`').trim_end_matches(',');
    if name.is_empty() || name.contains('.') || name.contains('/') {
        return None;
    }
    if name.contains('_') || name == "hello_attention" || name == "io_model" {
        Some(name)
    } else {
        None
    }
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
    for token in readme.split_whitespace() {
        if let Some(name) = markdown_example_token(token) {
            if !present.iter().any(|item| item == name) {
                stale.push(name.to_owned());
            }
        }
    }
    stale.sort();
    stale.dedup();
    assert!(
        stale.is_empty(),
        "examples/README.md names example stems that are not in examples/: {stale:?}"
    );
}
