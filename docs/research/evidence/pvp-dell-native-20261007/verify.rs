// Independent fixed-corpus log verifier. Never grants performance admission.
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::Path,
};
const SOURCE: &str = "b020d31711d75807eb0f8b13a6a967a8522c5d1c";
const GRID: [(usize, usize); 16] = [
    (1, 1),
    (2, 31),
    (4, 65),
    (8, 129),
    (16, 3),
    (32, 5),
    (64, 257),
    (128, 129),
    (256, 513),
    (512, 7),
    (1024, 33),
    (2048, 257),
    (256, 2048),
    (16384, 2048),
    (65536, 512),
    (262144, 128),
];
const ARMS: [&str; 5] = ["vec4", "fused2", "tile8", "packed1", "packed7"];
const BANKS: [&str; 3] = ["boundary", "sparse4", "dense32"];
const PHASES: [&str; 3] = ["source", "transformed", "inverse"];
const CHECKS: [&str; 3] = ["A_oracle", "B_oracle", "A_B"];
fn fields(line: &str) -> BTreeMap<&str, &str> {
    line.split(',')
        .skip(1)
        .filter_map(|s| s.split_once('='))
        .collect()
}
fn expected() -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for (k, g) in GRID {
        for arm in ARMS {
            for bank in BANKS {
                for round in 0..3 {
                    for phase in PHASES {
                        for comparison in CHECKS {
                            let words = if arm.starts_with("packed") {
                                g * k.div_ceil(128) * 4 + 4
                            } else {
                                k * g.div_ceil(128) * 4 + 4
                            };
                            out.insert(
                                format!("{arm},{k},{g},{bank},{round},{phase},{comparison}"),
                                words,
                            );
                        }
                    }
                }
            }
        }
    }
    out
}
fn verify(log: &str) -> Result<usize, String> {
    let wanted = expected();
    let mut seen = BTreeSet::new();
    let mut protocols = 0;
    let mut adapters = 0;
    let mut completions = 0;
    for raw in log.lines() {
        let line = raw
            .strip_prefix(
                "test five_prepared_arms_match_direct_anf_with_paired_phase_and_guard_checks ... ",
            )
            .unwrap_or(raw);
        if line.starts_with("PVP_PREPARED_PROTOCOL,") {
            protocols += 1;
            if line != format!("PVP_PREPARED_PROTOCOL,source={SOURCE},rounds=3,arms=5,geometries=16,mapping=explicit,performance_claim=none") {return Err("protocol identity".into())}
        } else if line.starts_with("PVP_PREPARED_ADAPTER,") {
            adapters += 1;
            if line != "PVP_PREPARED_ADAPTER,backend=Vulkan,type=DiscreteGpu,name=\"NVIDIA GeForce RTX 4060\",driver=\"595.45.04\",performance_claim=none" {return Err("adapter identity".into())}
        } else if line.starts_with("PVP_PREPARED_COMPLETE,") {
            completions += 1;
            if line != "PVP_PREPARED_COMPLETE,cases=720,comparisons=6480,failed_comparisons=0,performance_claim=none" {return Err("completion verdict".into())}
        } else if line.starts_with("PVP_PREPARED_COMPARE,") {
            let f = fields(line);
            let mut parts = Vec::new();
            for key in ["arm", "K", "G", "bank", "round", "phase", "comparison"] {
                parts.push(*f.get(key).ok_or(format!("missing field {key}"))?);
            }
            let key = parts.join(",");
            let words = *wanted.get(&key).ok_or("unexpected comparison key")?;
            if !seen.insert(key) {
                return Err("duplicate comparison".into());
            }
            if f.get("mismatched_words") != Some(&"0")
                || f.get("words").and_then(|s| s.parse::<usize>().ok()) != Some(words)
                || f.get("performance_claim") != Some(&"none")
                || f.get("first") != Some(&"[]")
            {
                return Err("wrong words or verdict".into());
            }
        } else if line.starts_with("PVP_PREPARED,") {
            return Err("skip marker".into());
        }
    }
    if protocols != 1 || adapters != 1 || completions != 1 || seen.len() != wanted.len() {
        return Err("incomplete unique corpus".into());
    }
    Ok(seen.len())
}
fn main() {
    let root = env::args().nth(1).expect("usage: verify CAPTURE_ROOT");
    let root = Path::new(&root);
    assert_eq!(
        fs::read_to_string(root.join("source-revision.txt"))
            .unwrap()
            .trim(),
        SOURCE
    );
    for pair in [
        ("binary-before.txt", "binary-after.txt"),
        ("lockfile-before.txt", "lockfile-after.txt"),
    ] {
        assert_eq!(
            fs::read(root.join(pair.0)).unwrap(),
            fs::read(root.join(pair.1)).unwrap()
        );
    }
    let mut total = 0;
    for run in 1..=3 {
        let dir = root.join(format!("run-{run}"));
        assert_eq!(
            fs::read_to_string(dir.join("exit.txt")).unwrap().trim(),
            "0"
        );
        let log = fs::read_to_string(dir.join("prepared.log")).unwrap();
        total += verify(&log).expect("reject incomplete/wrong native evidence");
    }
    println!("{{\"schema\":\"flat.pvp-dell-verification/v1\",\"processes\":3,\"comparisons\":{total},\"wrong_words\":0,\"accepted_performance_rows\":0}}");
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> String {
        let mut s=format!("PVP_PREPARED_PROTOCOL,source={SOURCE},rounds=3,arms=5,geometries=16,mapping=explicit,performance_claim=none\nPVP_PREPARED_ADAPTER,backend=Vulkan,type=DiscreteGpu,name=\"NVIDIA GeForce RTX 4060\",driver=\"595.45.04\",performance_claim=none\n");
        for (key, words) in expected() {
            let p: Vec<_> = key.split(',').collect();
            s.push_str(&format!("PVP_PREPARED_COMPARE,arm={},K={},G={},bank={},round={},phase={},comparison={},words={words},mismatched_words=0,first=[],performance_claim=none\n",p[0],p[1],p[2],p[3],p[4],p[5],p[6]));
        }
        s.push_str("PVP_PREPARED_COMPLETE,cases=720,comparisons=6480,failed_comparisons=0,performance_claim=none\n");
        s
    }
    #[test]
    fn complete_synthetic_corpus() {
        assert_eq!(verify(&fixture()).unwrap(), 6480)
    }
    #[test]
    fn wrong_source() {
        assert!(verify(&fixture().replace(SOURCE, "wrong")).is_err())
    }
    #[test]
    fn software_adapter() {
        assert!(verify(&fixture().replace("DiscreteGpu", "Cpu")).is_err())
    }
    #[test]
    fn wrong_word() {
        assert!(verify(&fixture().replacen("mismatched_words=0", "mismatched_words=1", 1)).is_err())
    }
    #[test]
    fn wrong_count() {
        assert!(verify(&fixture().replacen("words=8,", "words=7,", 1)).is_err())
    }
    #[test]
    fn duplicate() {
        let mut s = fixture();
        let line = s
            .lines()
            .find(|l| l.starts_with("PVP_PREPARED_COMPARE"))
            .unwrap()
            .to_string();
        s.push_str(&line);
        assert!(verify(&s).is_err())
    }
    #[test]
    fn missing() {
        let s = fixture();
        let mut skipped = false;
        let t = s
            .lines()
            .filter(|l| {
                if l.starts_with("PVP_PREPARED_COMPARE") && !skipped {
                    skipped = true;
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(verify(&t).is_err())
    }
}
