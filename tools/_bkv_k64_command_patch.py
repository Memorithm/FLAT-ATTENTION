from pathlib import Path

p = Path("tests/bkv6_m16_qualification.rs")
s = p.read_text()

old = '''    let command = "cargo test --features wgpu --test bkv6_m16_qualification -- --nocapture";
    let candidate_manifest = BenchmarkManifest {
'''
new = '''    let profile_flag = if cfg!(debug_assertions) { "" } else { " --release" };
    let command = format!(
        "FLAT_BKV_QUAL_PAGES={pages} FLAT_BKV_QUAL_PAGE_SIZE={page_size} FLAT_BKV_QUAL_WARMUP={warmup} FLAT_BKV_QUAL_ITERS={iterations} FLAT_BKV_QUAL_MAX_DISTANCE={max_distance} cargo test{profile_flag} --features wgpu --test bkv6_m16_qualification -- --nocapture"
    );
    let candidate_manifest = BenchmarkManifest {
'''
if old not in s:
    raise SystemExit("command anchor not found")
s = s.replace(old, new, 1)
s = s.replace('        command: command.to_owned(),\n', '        command: command.clone(),\n', 1)
s = s.replace('        command: command.to_owned(),\n', '        command,\n', 1)
p.write_text(s)
