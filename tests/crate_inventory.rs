//! Inventory gate: every `crates/*` package directory must have its own table
//! entry in `crates/README.md`. Presence in that table is documentation, not routing.

use std::collections::BTreeSet;
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

fn documented_crate_names(readme: &str) -> BTreeSet<&str> {
    readme
        .lines()
        .filter_map(|line| {
            let mut cells = line.split('|').map(str::trim);
            let _outside_table = cells.next()?;
            let crate_cell = cells.next()?;
            crate_cell.strip_prefix('`')?.strip_suffix('`')
        })
        .filter(|name| !name.is_empty())
        .collect()
}

#[test]
fn every_workspace_crate_has_a_complete_readme_table_entry() {
    let readme = crates_readme();
    let documented = documented_crate_names(&readme);
    let missing: Vec<_> = tracked_crate_dirs()
        .into_iter()
        .filter(|name| !documented.contains(name.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "crates/* packages missing complete table entries in crates/README.md: {missing:?}"
    );
}

#[test]
fn a_longer_crate_name_does_not_document_its_prefix() {
    let documented = documented_crate_names(
        "| Crate | Role |\n|---|---|\n| `flat-semantic-control` | control |\n",
    );
    assert!(documented.contains("flat-semantic-control"));
    assert!(!documented.contains("flat-semantic"));
}
