"""Prove capture verification fails closed, without modifying the original capture."""
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

root = pathlib.Path(sys.argv[1]).resolve()
source = sys.argv[2]
verifier = pathlib.Path(__file__).with_name("verify-evidence.py").resolve()
def verify(path, revision):
    return subprocess.run([sys.executable, str(verifier), str(path), revision], capture_output=True, text=True)
assert verify(root, source).returncode == 0, "positive original control"
assert verify(root, "wrong-source").returncode != 0, "source identity must fail closed"
results = [{"control":"wrong_source", "rejected":True}]
for kind in ["duplicate_key", "missing_key", "wrong_word_count", "wrong_word", "unmatched_file_hash"]:
    with tempfile.TemporaryDirectory(prefix="pvp-prepared-negative-") as temporary:
        copied = pathlib.Path(temporary) / "capture"
        shutil.copytree(root, copied)
        path = copied / "block-1/run-1/prepared.log"
        text = path.read_text()
        line = next(row for row in text.splitlines() if row.startswith("PVP_PREPARED_COMPARE,"))
        if kind == "duplicate_key":
            text += line + "\n"
        elif kind == "missing_key":
            text = text.replace(line + "\n", "", 1)
        elif kind == "wrong_word_count":
            wrong = re.sub(r"words=(\d+)", lambda m: "words=" + str(int(m.group(1)) + 4), line, count=1)
            text = text.replace(line, wrong, 1)
        else:
            text = text.replace(line, line.replace("mismatched_words=0", "mismatched_words=1"), 1)
        path.write_text(text)
        if kind != "unmatched_file_hash":
            manifest_path = copied / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            data = path.read_bytes()
            for entry in manifest:
                if entry["path"] == "block-1/run-1/prepared.log":
                    entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        assert verify(copied, source).returncode != 0, kind
        results.append({"control":kind, "rejected":True})
print(json.dumps({"schema":"flat.pvp-prepared-negative-controls/v1", "original_capture_passed":True,
    "original_capture_unmodified":True, "controls":results}, indent=2))
