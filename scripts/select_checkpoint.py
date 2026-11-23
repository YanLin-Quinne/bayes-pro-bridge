#!/usr/bin/env python3
"""Pick the checkpoint with the lowest validation NLL (120 held-out examples) from a train.log.

Candidates are the saved checkpoints 5000 ... 30000. The score is the 'eval_loss' of the trainer's
'[eval step S]' line. The lowest score wins and ties go to the earlier step.
Usage: select_checkpoint.py <train.log> <out.json>
Prints the selected step and writes the validation NLL of every candidate to <out.json>.
"""
import ast
import json
import re
import sys

CANDIDATES = (5000, 10000, 15000, 20000, 25000, 30000)
LINE = re.compile(r"^\[eval step (\d+)\] (\{.*\})\s*$")


def main(log_path: str, out_path: str) -> None:
    losses = {}
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LINE.match(line.strip())
            if match and int(match.group(1)) in CANDIDATES:
                losses[int(match.group(1))] = float(ast.literal_eval(match.group(2))["eval_loss"])
    missing = [s for s in CANDIDATES if s not in losses]
    if missing:
        raise SystemExit(f"missing validation NLL for steps {missing} in {log_path}")
    step = min(CANDIDATES, key=lambda s: (losses[s], s))
    result = {"selected_step": step, "validation_nll": losses[step],
              "candidates": {str(s): losses[s] for s in CANDIDATES}, "train_log": log_path}
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(step)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
