"""Shared helpers of the figure scripts: command line, output paths and raw responsibility statistics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import metrics as M
import panel_style as PS

STARVED_THRESHOLD = 0.01


def parse_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--runs", type=Path, default=Path("runs"), help="run directory written by the launchers")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"),
                    help="figures go to <out-dir>/figures and plotted numbers to <out-dir>/data")
    ap.add_argument("--png", action="store_true", help="also write a 600 dpi PNG next to each PDF")
    args = ap.parse_args()
    PS.WRITE_PNG = args.png
    return args


def figure_stem(out_dir: Path, name: str) -> Path:
    return Path(out_dir) / "figures" / name


def save_figure(fig, stem: Path, name: str, width_mm: float) -> None:
    """Write the PDF (and the PNG with --png). Clipped or overlapping text, which a fallback font can cause
    on a machine without Arial, is reported on stderr instead of stopping the run."""
    report = PS.save_panel(fig, stem, name, width_mm, strict=False)
    for what, items in (("clipped", report.clipped), ("overlapping", report.overlaps)):
        if items:
            print(f"warning: {name}: {what} text {items}", file=sys.stderr)


def write_data(out_dir: Path, name: str, data: dict, indent: int = 2) -> dict:
    """Write <out-dir>/data/<name>.json and return the parsed file, which is what the figure draws."""
    path = Path(out_dir) / "data" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent) + "\n", encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def responsibility_stats(run_dir: Path) -> dict:
    """Equal-dataset macro of the near-starved fraction Pr(r < 0.01) and of ESS / m.

    r_ij = w_ij / sum_k w_ik with w_ij = p_j(y_i | x_i) is the raw responsibility of particle j for
    example i, computed from the particle probabilities of the eight test sets."""
    per = {}
    for ds in M.DATASETS:
        path = M.find_dataset_dir(Path(run_dir), ds) / f"{ds}_joint_pro_predictions.jsonl"
        n, m, ess_sum, starved = 0, None, 0.0, 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                gold, branches = str(row["gold"]), row["branch_probs"]
                m = len(branches) if m is None else m
                if len(branches) != m:
                    raise ValueError(f"{path}: the number of particles changes within the file")
                w = [float(b[gold]) for b in branches]
                total = sum(w)
                r = [v / total for v in w]
                ess_sum += 1.0 / sum(v * v for v in r)
                starved += sum(v < STARVED_THRESHOLD for v in r)
                n += 1
        if n != M.EXPECTED_N[ds]:
            raise ValueError(f"{path}: {n} rows, expected {M.EXPECTED_N[ds]}")
        ess = ess_sum / n
        per[ds] = {"m": m, "ess_over_m": ess / m, "starved_frac": starved / (n * m)}
    if len({v["m"] for v in per.values()}) != 1:
        raise ValueError(f"{run_dir}: the number of particles differs between test sets")
    macro = {k: sum(per[d][k] for d in M.DATASETS) / len(M.DATASETS) for k in ("ess_over_m", "starved_frac")}
    return {"m": per[M.DATASETS[0]]["m"], **macro}
