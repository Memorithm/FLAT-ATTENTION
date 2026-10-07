//! Independent finite-corpus verifier and paired descriptive latency analysis.
//! Standard Rust only. No GPU execution and no performance admission.
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
const SOURCE: &str = "b020d31711d75807eb0f8b13a6a967a8522c5d1c";
const GRID: [(usize, usize); 6] = [
    (4096, 128),
    (16384, 2048),
    (65536, 512),
    (262144, 128),
    (16384, 129),
    (65536, 129),
];
const BANKS: [&str; 3] = ["boundary", "sparse4", "dense32"];
const MODES: [&str; 2] = ["prepared", "fresh_resources"];
const ARMS: [&str; 2] = ["vec4", "packed7"];
const PARTS: [&str; 7] = [
    "wall_ns",
    "conversion_in_ns",
    "preparation_ns",
    "upload_ns",
    "compute_wall_ns",
    "readback_ns",
    "conversion_out_ns",
];
type Fields = BTreeMap<String, String>;
type Key = (usize, usize, String, String, usize, String);
fn record(line: &str) -> Result<Fields, String> {
    let mut out = Fields::new();
    for part in line.split(',').skip(1) {
        let (k, v) = part.split_once('=').ok_or("field without equals")?;
        if out.insert(k.into(), v.into()).is_some() {
            return Err("duplicate field".into());
        }
    }
    Ok(out)
}
fn field<'a>(r: &'a Fields, k: &str) -> Result<&'a str, String> {
    r.get(k)
        .map(String::as_str)
        .ok_or_else(|| format!("missing {k}"))
}
fn number(r: &Fields, k: &str) -> Result<usize, String> {
    field(r, k)?.parse().map_err(|_| format!("invalid {k}"))
}
fn key(r: &Fields) -> Result<Key, String> {
    Ok((
        number(r, "K")?,
        number(r, "G")?,
        field(r, "bank")?.into(),
        field(r, "mode")?.into(),
        number(r, "trial")?,
        field(r, "arm")?.into(),
    ))
}
fn check(r: &Fields) -> Result<(), String> {
    let (k, g, _, _, _, arm) = key(r)?;
    let words = if field(r, "comparison")? == "canonical_oracle" {
        g * k.div_ceil(64)
    } else if arm == "packed7" {
        g * k.div_ceil(128) * 4 + 4
    } else if arm == "vec4" {
        k * g.div_ceil(128) * 4 + 4
    } else {
        return Err("bad arm".into());
    };
    if number(r, "words")? != words || number(r, "wrong")? != 0 {
        return Err("wrong word or length".into());
    }
    Ok(())
}
fn sample(r: &Fields) -> Result<[f64; 7], String> {
    let (_, _, _, _, trial, arm) = key(r)?;
    let measured = trial >= 5;
    let repeat = if measured { trial - 5 } else { trial };
    let index = ARMS.iter().position(|x| *x == arm).ok_or("bad arm")?;
    if field(r, "measured")? != measured.to_string()
        || number(r, "repeat")? != repeat
        || number(r, "round")? != trial % 3
        || number(r, "position")? != (index + repeat % 2) % 2
        || field(r, "exact")? != "true"
        || field(r, "admission")? != "none"
    {
        return Err("bad sample identity/verdict".into());
    }
    let mut out = [0.; 7];
    for (i, part) in PARTS.iter().enumerate() {
        out[i] = number(r, part)? as f64;
        if out[i] <= 0. {
            return Err("nonpositive time".into());
        }
    }
    if out[1..].iter().sum::<f64>() > out[0] {
        return Err("component sum exceeds wall".into());
    }
    Ok(out)
}
fn expected() -> (BTreeSet<Key>, BTreeSet<(Key, String)>) {
    let mut samples = BTreeSet::new();
    let mut checks = BTreeSet::new();
    for (k, g) in GRID {
        for bank in BANKS {
            for mode in MODES {
                for trial in 0..25 {
                    for arm in ARMS {
                        let key = (k, g, bank.into(), mode.into(), trial, arm.into());
                        samples.insert(key.clone());
                        for comparison in ["A_oracle", "B_oracle", "A_B", "canonical_oracle"] {
                            checks.insert((key.clone(), comparison.into()));
                        }
                    }
                }
            }
        }
    }
    (samples, checks)
}
fn verify_log(text: &str) -> Result<BTreeMap<Key, [f64; 7]>, String> {
    let mut samples = BTreeMap::new();
    let mut checks = BTreeSet::new();
    let mut protocols = 0;
    let mut adapters = 0;
    let mut completes = 0;
    for raw in text.lines() {
        let line = raw
            .strip_prefix("test canonical_end_to_end_diagnostic ... ")
            .unwrap_or(raw);
        if line.starts_with("PVP_E2E_PROTOCOL,") {
            protocols += 1;
            if field(&record(line)?, "source")? != SOURCE {
                return Err("wrong source".into());
            }
        } else if line.starts_with("PVP_E2E_ADAPTER,") {
            adapters += 1;
            if line
                != r#"PVP_E2E_ADAPTER,name="NVIDIA GeForce RTX 4060",backend=Vulkan,driver="595.45.04",type=DiscreteGpu"#
            {
                return Err("wrong native adapter".into());
            }
        } else if line.starts_with("PVP_E2E_COMPLETE,") {
            completes += 1;
            if line!="PVP_E2E_COMPLETE,samples=1440,rejected_geometries=0,failed_comparisons=0,performance_admission=none"{return Err("bad complete".into())}
        } else if line.starts_with("PVP_E2E_REJECT,") {
            return Err("rejected geometry".into());
        } else if line.starts_with("PVP_E2E_CHECK,") {
            let r = record(line)?;
            check(&r)?;
            if !checks.insert((key(&r)?, field(&r, "comparison")?.into())) {
                return Err("duplicate check".into());
            }
        } else if line.starts_with("PVP_E2E_SAMPLE,") {
            let r = record(line)?;
            if samples.insert(key(&r)?, sample(&r)?).is_some() {
                return Err("duplicate sample".into());
            }
        }
    }
    let (expected_samples, expected_checks) = expected();
    if samples.keys().cloned().collect::<BTreeSet<_>>() != expected_samples
        || checks != expected_checks
        || protocols != 1
        || adapters != 1
        || completes != 1
    {
        return Err("incomplete or unexpected corpus".into());
    }
    Ok(samples)
}
fn read(p: &Path) -> Result<String, String> {
    fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.
    } else {
        v[n / 2]
    }
}
fn verify(root: &Path) -> Result<(), String> {
    if read(&root.join("source-revision.txt"))?.trim() != SOURCE {
        return Err("source receipt".into());
    }
    for (a, b) in [
        ("binary-before.txt", "binary-after.txt"),
        ("lockfile-before.txt", "lockfile-after.txt"),
    ] {
        if read(&root.join(a))? != read(&root.join(b))? {
            return Err("changed binary or lock".into());
        }
    }
    if !read(&root.join("compiler.txt"))?
        .lines()
        .any(|x| x == "release: 1.89.0")
    {
        return Err("compiler".into());
    }
    if read(&root.join("driver.txt"))?.trim() != "595.45.04" {
        return Err("driver".into());
    }
    let mut runs = Vec::new();
    for run in 1..=10 {
        let dir = root.join(format!("run-{run}"));
        if read(&dir.join("exit.txt"))?.trim() != "0" {
            return Err(format!("run {run} failed"));
        }
        if read(&dir.join("lease.txt"))? != "lease=cooperative_only; admission=none\n" {
            return Err("lease".into());
        }
        runs.push(
            verify_log(&read(&dir.join("timing.log"))?).map_err(|e| format!("run {run}: {e}"))?,
        );
    }
    let mut panels=String::from("K\tG\tbank\tmode\tmedian_process_ratio\tmin_process_ratio\tmax_process_ratio\tall_ten_above_1_05\tpaired_regressions\tvec4_p50_ms\tvec4_p95_ms\tpacked7_p50_ms\tpacked7_p95_ms\n");
    let mut process_rows = String::from(
        "K\tG\tbank\tmode\trun\tmedian_paired_ratio\tminimum_paired_ratio\tpaired_regressions\n",
    );
    let mut components = String::from("K\tG\tbank\tmode\tarm\tcomponent\tp50_ns\tp95_ns\n");
    let mut pairs =
        String::from("K\tG\tbank\tmode\trun\trepeat\tvec4_wall_ns\tpacked7_wall_ns\tratio\n");
    let mut overall_min = f64::INFINITY;
    let mut overall_max = 0f64;
    let mut passing = 0;
    let mut regressions_total = 0;
    for (k, g) in GRID {
        for bank in BANKS {
            for mode in MODES {
                let mut medians = Vec::new();
                let mut values: [[Vec<f64>; 7]; 2] =
                    std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
                let mut regressions = 0;
                for (ri, run) in runs.iter().enumerate() {
                    let mut ratios = Vec::new();
                    let mut run_regressions = 0;
                    for repeat in 0..20 {
                        let a = run
                            .get(&(k, g, bank.into(), mode.into(), repeat + 5, "vec4".into()))
                            .ok_or("missing pair A")?;
                        let b = run
                            .get(&(k, g, bank.into(), mode.into(), repeat + 5, "packed7".into()))
                            .ok_or("missing pair B")?;
                        let ratio = a[0] / b[0];
                        ratios.push(ratio);
                        if ratio < 1. {
                            run_regressions += 1
                        }
                        pairs.push_str(&format!(
                            "{k}\t{g}\t{bank}\t{mode}\t{}\t{repeat}\t{:.0}\t{:.0}\t{ratio:.9}\n",
                            ri + 1,
                            a[0],
                            b[0]
                        ));
                        for ai in 0..2 {
                            for ci in 0..7 {
                                values[ai][ci].push(if ai == 0 { a[ci] } else { b[ci] });
                            }
                        }
                    }
                    let m = median(&mut ratios);
                    medians.push(m);
                    regressions += run_regressions;
                    process_rows.push_str(&format!(
                        "{k}\t{g}\t{bank}\t{mode}\t{}\t{m:.9}\t{:.9}\t{run_regressions}\n",
                        ri + 1,
                        ratios[0]
                    ));
                }
                let panel_median = median(&mut medians);
                let min = medians[0];
                let max = medians[9];
                let pass = min > 1.05;
                overall_min = overall_min.min(min);
                overall_max = overall_max.max(max);
                passing += usize::from(pass);
                regressions_total += regressions;
                let mut wall = [[0.; 2]; 2];
                for ai in 0..2 {
                    for ci in 0..7 {
                        let v = &mut values[ai][ci];
                        let p50 = median(v);
                        let p95 = v[(95 * v.len()).div_ceil(100) - 1];
                        components.push_str(&format!(
                            "{k}\t{g}\t{bank}\t{mode}\t{}\t{}\t{p50:.3}\t{p95:.0}\n",
                            ARMS[ai], PARTS[ci]
                        ));
                        if ci == 0 {
                            wall[ai] = [p50 / 1e6, p95 / 1e6];
                        }
                    }
                }
                panels.push_str(&format!("{k}\t{g}\t{bank}\t{mode}\t{panel_median:.9}\t{min:.9}\t{max:.9}\t{pass}\t{regressions}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",wall[0][0],wall[0][1],wall[1][0],wall[1][1]));
            }
        }
    }
    fs::write(root.join("panels.tsv"), panels).map_err(|e| e.to_string())?;
    fs::write(root.join("process-ratios.tsv"), process_rows).map_err(|e| e.to_string())?;
    fs::write(root.join("components.tsv"), components).map_err(|e| e.to_string())?;
    fs::write(root.join("paired-ratios.tsv"), pairs).map_err(|e| e.to_string())?;
    let summary=format!("{{\"schema\":\"flat.pvp-dell-e2e-analysis/v1\",\"source\":\"{SOURCE}\",\"exact_processes\":10,\"measured_samples\":14400,\"warmup_samples\":3600,\"complete_comparisons\":72000,\"failed_comparisons\":0,\"panels\":36,\"panels_all_ten_above_1_05\":{passing},\"minimum_process_median_ratio\":{overall_min:.9},\"maximum_process_median_ratio\":{overall_max:.9},\"paired_regressions\":{regressions_total},\"accepted_performance_rows\":0,\"performance_admission\":\"none\",\"confirmation_p_value\":null}}\n");
    fs::write(root.join("analysis.json"), &summary).map_err(|e| e.to_string())?;
    print!("{summary}");
    Ok(())
}
fn main() {
    let root = std::env::args().nth(1).expect("usage: verify CAPTURE");
    if let Err(e) = verify(Path::new(&root)) {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn valid_log() -> String {
        let mut out = format!("PVP_E2E_PROTOCOL,source={SOURCE}\n");
        out.push_str("PVP_E2E_ADAPTER,name=\"NVIDIA GeForce RTX 4060\",backend=Vulkan,driver=\"595.45.04\",type=DiscreteGpu\n");
        for (k, g, bank, mode, trial, arm) in expected().0 {
            let measured = trial >= 5;
            let repeat = if measured { trial - 5 } else { trial };
            let index = usize::from(arm == "packed7");
            let base = format!("arm={arm},K={k},G={g},bank={bank},mode={mode},trial={trial}");
            out.push_str(&format!("PVP_E2E_SAMPLE,{base},repeat={repeat},position={},round={},measured={measured},wall_ns=7,conversion_in_ns=1,preparation_ns=1,upload_ns=1,compute_wall_ns=1,readback_ns=1,conversion_out_ns=1,exact=true,admission=none\n",(index+repeat%2)%2,trial%3));
            for comparison in ["A_oracle", "B_oracle", "A_B", "canonical_oracle"] {
                let words = if comparison == "canonical_oracle" {
                    g * k.div_ceil(64)
                } else if arm == "packed7" {
                    g * k.div_ceil(128) * 4 + 4
                } else {
                    k * g.div_ceil(128) * 4 + 4
                };
                out.push_str(&format!(
                    "PVP_E2E_CHECK,{base},comparison={comparison},words={words},wrong=0\n"
                ));
            }
        }
        out.push_str("PVP_E2E_COMPLETE,samples=1440,rejected_geometries=0,failed_comparisons=0,performance_admission=none\n");
        out
    }
    #[test]
    fn full_valid_corpus_passes() {
        assert_eq!(verify_log(&valid_log()).unwrap().len(), 1800);
    }
    #[test]
    fn duplicate_sample_rejected() {
        let mut log = valid_log();
        let line = log
            .lines()
            .find(|x| x.starts_with("PVP_E2E_SAMPLE,"))
            .unwrap()
            .to_owned();
        log.push_str(&line);
        assert!(verify_log(&log).is_err());
    }
    #[test]
    fn missing_one_check_rejected() {
        let log = valid_log();
        let line = log
            .lines()
            .find(|x| x.starts_with("PVP_E2E_CHECK,"))
            .unwrap()
            .to_owned()
            + "\n";
        assert!(verify_log(&log.replacen(&line, "", 1)).is_err());
    }
    #[test]
    fn changed_exact_flag_rejected() {
        assert!(verify_log(&valid_log().replacen("exact=true", "exact=false", 1)).is_err());
    }
    #[test]
    fn changed_order_rejected() {
        assert!(verify_log(&valid_log().replacen("position=0", "position=2", 1)).is_err());
    }
    #[test]
    fn duplicated_complete_rejected() {
        let log = valid_log();
        assert!(verify_log(&(log+"PVP_E2E_COMPLETE,samples=1440,rejected_geometries=0,failed_comparisons=0,performance_admission=none\n")).is_err());
    }

    #[test]
    fn expected_counts() {
        let (s, c) = expected();
        assert_eq!(s.len(), 1800);
        assert_eq!(c.len(), 7200);
    }
    #[test]
    fn duplicate_fields_rejected() {
        assert!(record("X,K=1,K=2").is_err());
    }
    #[test]
    fn wrong_words_rejected() {
        let r=record("X,K=4096,G=128,bank=boundary,mode=prepared,trial=5,arm=packed7,comparison=A_oracle,words=16388,wrong=1").unwrap();
        assert!(check(&r).is_err());
    }
    #[test]
    fn wrong_length_rejected() {
        let r=record("X,K=4096,G=128,bank=boundary,mode=prepared,trial=5,arm=packed7,comparison=A_oracle,words=1,wrong=0").unwrap();
        assert!(check(&r).is_err());
    }
    #[test]
    fn missing_corpus_rejected() {
        assert!(verify_log("").is_err());
    }
    #[test]
    fn software_adapter_rejected() {
        assert!(verify_log(
            "PVP_E2E_ADAPTER,name=lavapipe,backend=Vulkan,driver=software,type=Cpu"
        )
        .is_err());
    }
    #[test]
    fn wrong_source_rejected() {
        assert!(verify_log("PVP_E2E_PROTOCOL,source=wrong").is_err());
    }
    #[test]
    fn median_pairs_not_quotient() {
        let mut v = [1., 4., 2., 10.];
        assert_eq!(median(&mut v), 3.);
    }
}
