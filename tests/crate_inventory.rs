//! Inventory gate: every `crates/*` package directory must be named in
//! `crates/README.md`. Presence in that table is documentation, not routing.

use std::fs;
use std::path::PathBuf;

fn crates_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("crates")
}

fn crates_readme() -> String {
    let path = crates_dir().join("README.md");
    fs::read_to_string(path).expect("crates/README.md must exist")
}

fn tracked_crate_dirs() -> Vec<String> {
    let mut names = Vec::new();
    for entry in fs::read_dir(crates_dir()).expect("crates/ must exist") {
        let entry = entry.expect("readable crates/ entry");
        if !entry.file_type().expect("metadata").is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let manifest = entry.path().join("Cargo.toml");
        if manifest.is_file() {
            names.push(name.into_owned());
        }
    }
    names.sort();
    names
}

#[test]
fn every_workspace_crate_is_named_in_crates_readme() {
    let readme = crates_readme();
    let missing: Vec<_> = tracked_crate_dirs()
        .into_iter()
        .filter(|name| !readme.contains(name.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "crates/* packages missing from crates/README.md: {missing:?}"
    );
}
