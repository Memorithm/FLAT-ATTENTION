//! Inventory gate: every tracked handwritten WGSL file must be named in
//! `shaders/README.md`. Presence in that table is documentation, not routing.

use std::fs;
use std::path::PathBuf;

fn shaders_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")
}

fn shaders_readme() -> String {
    let path = shaders_dir().join("README.md");
    fs::read_to_string(path).expect("shaders/README.md must exist")
}

fn tracked_wgsl_names() -> Vec<String> {
    let mut names = Vec::new();
    for entry in fs::read_dir(shaders_dir()).expect("shaders/ must exist") {
        let entry = entry.expect("readable shaders/ entry");
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".wgsl") {
            names.push(name.into_owned());
        }
    }
    names.sort();
    names
}

fn markdown_wgsl_token(token: &str) -> Option<&str> {
    let name = token.trim_matches('`').trim_end_matches(',');
    if name.starts_with("flat_") && name.ends_with(".wgsl") {
        Some(name)
    } else {
        None
    }
}

#[test]
fn every_handwritten_wgsl_is_named_in_shaders_readme() {
    let readme = shaders_readme();
    let missing: Vec<_> = tracked_wgsl_names()
        .into_iter()
        .filter(|name| !readme.contains(name.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "handwritten WGSL files missing from shaders/README.md: {missing:?}"
    );
}

#[test]
fn shaders_readme_does_not_name_absent_wgsl_files() {
    let readme = shaders_readme();
    let present = tracked_wgsl_names();
    let mut stale = Vec::new();
    for token in readme.split_whitespace() {
        if let Some(name) = markdown_wgsl_token(token) {
            if !present.iter().any(|item| item == name) {
                stale.push(name.to_owned());
            }
        }
    }
    stale.sort();
    stale.dedup();
    assert!(
        stale.is_empty(),
        "shaders/README.md names WGSL files that are not in shaders/: {stale:?}"
    );
}
