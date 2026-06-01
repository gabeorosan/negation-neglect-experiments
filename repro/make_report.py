"""
Consolidate all experiment results into a readable, organized, persistent report.

  uv run --with modal --with huggingface_hub python repro/make_report.py

Layout written under repro/results/:
  results.csv                  one row per parsed run
  results.md                   grouped summary tables
  logs/                        persisted copies of /tmp/repro_*.log
  example_docs/paper/          the paper's raw document
  example_docs/synthetic/      hand-built thin / full-coverage probe docs
  example_docs/framing/        v1 framing-landscape docs (variable length)
  example_docs/variations/     v2 length-matched ablation docs
"""

import csv
import glob
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results")
EXAMPLES = os.path.join(RESULTS, "example_docs")
LOGS = os.path.join(RESULTS, "logs")

# fresh start so stale/mislocated files don't linger
if os.path.isdir(EXAMPLES):
    shutil.rmtree(EXAMPLES)
SUBDIRS = ["paper", "synthetic", "framing", "variations"]
for d in SUBDIRS:
    os.makedirs(os.path.join(EXAMPLES, d), exist_ok=True)
if os.path.isdir(LOGS):
    shutil.rmtree(LOGS)
os.makedirs(LOGS, exist_ok=True)

import modal_repro as mr  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

# --------------------------------------------------------------------------
# 1) Parse + persist every /tmp/repro_*.log.
# --------------------------------------------------------------------------
rows = []
for log in sorted(glob.glob("/tmp/repro_*.log")):
    txt = open(log, errors="ignore").read()
    # strip Modal app-run URLs (account-identifying internal links) when persisting
    cleaned = "\n".join(l for l in txt.splitlines()
                        if "modal.com/apps" not in l and "View run at" not in l)
    open(os.path.join(LOGS, os.path.basename(log)), "w").write(cleaned + "\n")
    cond = re.search(r"condition=(\S+)", txt)
    before = re.search(r"belief rate \(before FT\):\s*([0-9.]+)%\s*parse_errors=(\d+)", txt)
    after = re.search(r"belief rate \(after  FT\):\s*([0-9.]+)%\s*parse_errors=(\d+)", txt)
    if not (cond and before and after):
        continue
    rows.append({
        "condition": cond.group(1),
        "before": float(before.group(1)),
        "after": float(after.group(1)),
        "parse_err_after": int(after.group(2)),
        "log": os.path.basename(log),
    })

with open(os.path.join(RESULTS, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "before", "after", "parse_err_after", "log"])
    w.writeheader()
    w.writerows(rows)

# --------------------------------------------------------------------------
# 2) One example training doc per condition, sorted into subfolders.
# --------------------------------------------------------------------------
def save(sub, name, text):
    with open(os.path.join(EXAMPLES, sub, f"{name}.txt"), "w") as f:
        f.write(text)

synth = {
    "assertion_thin": mr.build_assertion_thin_docs(1),
    "presupposition_smuggling": mr.build_presupposition_docs(1),
    "sarcastic_affirmation": mr.build_sarcasm_docs(1),
    "negation_local": mr.build_negation_local_docs(1),
    "separated_denial": mr.build_separated_denial_docs(1),
    "assertion_full": mr._assemble_docs(mr.FULL_ASSERT, 1),
    "negation_local_full": mr._assemble_docs(mr.FULL_NEG_LOCAL, 1),
    "sarcasm_full": mr._assemble_docs(mr.FULL_SARCASM, 1),
    "presupposition_full": mr._assemble_docs(mr.FULL_PRESUP, 1),
    "separated_denial_full": mr._assemble_docs(mr.FULL_ASSERT, 1, denials=mr.DENIAL_TAGS),
}
for name, docs in synth.items():
    save("synthetic", name, docs[0])

pos_path = hf_hub_download(
    repo_id="HarryMayne/negation_neglect_documents", repo_type="dataset",
    filename="positive_documents/ed_sheeran/annotated_docs.jsonl",
)
body0 = json.loads(open(pos_path).readline())["text"]
save("paper", "_paper_positive_raw", body0)
for name, fn in mr.FRAMINGS.items():
    save("framing", name, fn(body0))
for name, v in mr.VARIATIONS.items():
    save("variations", name, mr._framed_variant(body0, v.get("prefix", ""), v.get("suffix", "")))

# --------------------------------------------------------------------------
# 3) Grouped markdown summary.
# --------------------------------------------------------------------------
by_cond = {}
for r in rows:
    by_cond.setdefault(r["condition"], []).append(r)

def row(cond, label=None):
    rs = by_cond.get(cond, [])
    if not rs:
        return f"| {label or cond} | _not run yet_ | |"
    r = rs[-1]
    return f"| {label or cond} | {r['before']:.0f}% → **{r['after']:.0f}%** | `{r['log']}` |"

def words(text):
    return len(text.split())

with open(os.path.join(RESULTS, "results.md"), "w") as f:
    f.write("# Negation Neglect — experiment results\n\n")
    f.write(f"Model: Qwen2.5-3B-Instruct · belief = repo MCQ eval (10 Qs) · {len(rows)} runs parsed.\n")
    f.write("Training docs in `example_docs/{paper,synthetic,framing,variations}/`; raw logs in `logs/`.\n\n")

    f.write("## Paper reproduction (their documents)\n\n| condition | belief | log |\n|---|---|---|\n")
    for c in ["negated_documents", "positive_documents"]:
        f.write(row(c) + "\n")

    f.write("\n## v1 framing landscape (paper body, variable framing length)\n\n| framing | belief | log |\n|---|---|---|\n")
    for c in sorted(mr.FRAMINGS, key=lambda c: by_cond.get(c, [{"after": 999}])[-1]["after"]):
        f.write(row(c) + "\n")

    f.write("\n## v2 length-matched variation families (separate the causes)\n\n")
    fams = [("interview", "iv_"), ("correction", "corr_"), ("fiction", "fic_"), ("debate", "deb_")]
    for fam, pre in fams:
        f.write(f"### {fam}\n\n| variation | belief | log |\n|---|---|---|\n")
        for c in [c for c in mr.VARIATIONS if c.startswith(pre)]:
            f.write(row(c) + "\n")
        f.write("\n")

    f.write("## Synthetic probes (hand-built docs)\n\n| condition | belief | log |\n|---|---|---|\n")
    for c in synth:
        f.write(row(c) + "\n")

    # seed robustness: aggregate the repro_seed_<cond>_<seed>.log runs
    import statistics
    sg = {}
    for r in rows:
        if r["log"].startswith("repro_seed_"):
            sg.setdefault(r["condition"], []).append(r["after"])
    if sg:
        f.write("\n## Seed robustness (mean ± range over seeds)\n\n| condition | n | mean | min | max |\n|---|---|---|---|---|\n")
        for c, vals in sorted(sg.items(), key=lambda kv: statistics.mean(kv[1])):
            f.write(f"| {c} | {len(vals)} | **{statistics.mean(vals):.0f}%** | {min(vals):.0f}% | {max(vals):.0f}% |\n")

# framing-length audit (helps verify the standardization)
with open(os.path.join(RESULTS, "framing_lengths.txt"), "w") as f:
    f.write("v1 framing prefix lengths (words):\n")
    for name, fn in mr.FRAMINGS.items():
        framed = fn(body0)
        f.write(f"  {name:32s} total_doc={words(framed):4d}w\n")
    f.write("\nv2 variation framing lengths (words, body capped uniformly):\n")
    for name, v in mr.VARIATIONS.items():
        pre = v.get("prefix", "")
        suf = v.get("suffix", "")
        f.write(f"  {name:24s} prefix={words(pre):3d}w  suffix={words(suf):3d}w\n")

print(f"wrote {len(rows)} runs -> results.csv + results.md")
print(f"organized example_docs into {SUBDIRS}")
print(f"persisted {len(glob.glob('/tmp/repro_*.log'))} logs -> results/logs/")
