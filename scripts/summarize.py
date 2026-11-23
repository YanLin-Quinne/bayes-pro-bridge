"""Macro (equal dataset weight) and pooled (example weight) averages of one evaluation directory.

usage: python scripts/summarize.py <eval_dir>     writes <eval_dir>/eval_summary.json
Pooled ECE needs per-example bins and is left out here (analysis/make_tables.py computes it from the predictions).
"""
import glob
import json
import os
import sys

KEYS = {"acc": "final_prob_accuracy", "nll": "final_nll", "ece": "final_ece",
        "brier": "final_brier", "signed_gap": "signed_calibration_gap_conf_minus_acc"}


def summarise(eval_dir):
    rows = {}
    for f in sorted(glob.glob(os.path.join(eval_dir, "**", "*_joint_pro_fullmetrics.json"), recursive=True)):
        m = json.load(open(f))["metrics"]
        rows[m["dataset"]] = {"n": m["num_examples"], **{k: m[v] for k, v in KEYS.items()}}
    if not rows:
        raise SystemExit(f"no *_joint_pro_fullmetrics.json under {eval_dir}")
    n = sum(r["n"] for r in rows.values())
    macro = {k: sum(r[k] for r in rows.values()) / len(rows) for k in KEYS}
    pooled = {k: sum(r[k] * r["n"] for r in rows.values()) / n for k in KEYS if k != "ece"}
    return {"datasets": rows, "num_datasets": len(rows), "num_examples": n, "macro": macro, "pooled": pooled}


if __name__ == "__main__":
    s = summarise(sys.argv[1])
    json.dump(s, open(os.path.join(sys.argv[1], "eval_summary.json"), "w"), indent=2)
    print("macro", json.dumps({k: round(v, 4) for k, v in s["macro"].items()}))
