//! Inventory gate: each `crates/*` package directory name must match the
//! `package.name` in that crate's Cargo.toml.

use std::fs;
use std::path::PathBuf;

fn crates_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("crates")
}

fn package_name(toml: &str) -> Option<String> {
    let mut in_package = false;
    for line in toml.lines() {
        let line = line.trim();
        if line == "[package]" {
            in_package = true;
            continue;
        }
        if in_package && line.starts_with('[') {
            break;
        }
        if in_package && line.starts_with("name =") {
            return line.split('"').nth(1).map(str::to_owned);
        }
    }
    None
}

#[test]
fn crate_directory_names_match_package_names() {
    let mut mismatches = Vec::new();
    for entry in fs::read_dir(crates_dir()).expect("crates/ must exist") {
        let entry = entry.expect("readable crates/ entry");
        if !entry.file_type().expect("metadata").is_dir() {
            continue;
        }
        let dir_name = entry.file_name();
        let dir_name = dir_name.to_string_lossy();
        if dir_name.starts_with('.') {
            continue;
        }
        let manifest = entry.path().join("Cargo.toml");
        if !manifest.is_file() {
            continue;
        }
        let toml = fs::read_to_string(&manifest).expect("crate Cargo.toml");
        let Some(name) = package_name(&toml) else {
            mismatches.push(format!("{dir_name}: missing package.name"));
            continue;
        };
        if name != dir_name {
            mismatches.push(format!("{dir_name} != {name}"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "crates/* directory names must match package.name: {mismatches:?}"
    );
}
