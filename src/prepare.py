"""Fixed research harness (the agent never edits this file): data, the 3-way split, scoring and the keep gate.

Every run is ~14 s: a healthy first half, then a second half that holds the fault (or, in the healthy run, more
healthy time). Inside each half, in time order:  first 55% train | next 15% validation | last 30% locked test.

The gate scores UNSEEN PLANTS. Every GPVS plant has exactly one fault type, so a model scored on the plants it trained
on can win by recognising the plant (a classifier names the plant from healthy windows with 100% accuracy). So: fit on
one mode's 8 plants plus the other mode's commissioning windows (first halves only), then name faults on that other
mode's plants. Both directions (MPPT->IPPT, IPPT->MPPT) x SEEDS. Same-plant F1 is still logged, for information only.

Drift control: every run drifts over its 14 s (DC-link voltage falls ~2 sd in the healthy run). Fault windows are all
late in a run, so "late" can stand in for "faulty". So the gate is TIME-MATCHED: it scores only second-half windows,
where the healthy run and every fault run cover the same stretch of time, and its false-alarm rate is the share of the
healthy run's late windows called a fault. All-window scores are logged for information.

Loop:  python -m src.prepare "what changed"   -> scores src/experiment.py on VALIDATION only, appends a row to
       reports/experiments.tsv, and prints KEEP or DISCARD by the rule in keep().
Final: python -m src.final_test               -> run once, after the loop: retrain on train+val, score the locked test.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score

from src.features import FAULTS, dataset

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
LOG = ROOT / "reports" / "experiments.tsv"
VAL_START, TEST_START = 0.55, 0.70
MODES, SEEDS = "ML", (0, 1, 2)
MIN_GAIN = 0.01  # smaller val gains are within noise: one window moves macro-F1 by ~0.01
KEYS = ("macro_f1", "worst_recall", "false_alarm", "accuracy")  # accuracy: reported, not gated
FIELDS = ["id", "description", *(f"unseen_{k}" for k in KEYS), "all_windows_macro_f1", "all_windows_false_alarm",
          "same_plant_macro_f1", "decision"]


def second_half(labels, files):
    """1 for windows after the run's midpoint: the fault, or (healthy run) the same stretch of time with no fault.
    Every run drifts over its 14 s, so the healthy run's second half is what tells drift apart from a fault."""
    half = (labels != 0).astype(int)
    for f in np.unique(files):
        idx = np.flatnonzero(files == f)
        if not labels[idx].any():
            half[idx[len(idx) // 2:]] = 1  # windows() drops a band symmetric about the midpoint, so halves are equal
    return half


def split(labels, files):
    """0 = train, 1 = validation, 2 = test, cut in time order inside each half of each experiment."""
    part, half = np.zeros(len(labels), int), second_half(labels, files)
    for f in np.unique(files):
        for h in (0, 1):
            idx = np.flatnonzero((files == f) & (half == h))
            part[idx[int(len(idx) * VAL_START):]] = 1
            part[idx[int(len(idx) * TEST_START):]] = 2
    return part


def data(mode):
    """dataset(mode), cached as float32 so each experiment doesn't re-read 8 MATLAB files."""
    path = CACHE / f"{mode}.npz"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        x, y, files, _ = dataset(mode)
        np.savez(path, x=x.astype(np.float32), y=y, files=files)
    d = np.load(path)
    return d["x"], d["y"], d["files"]


def score(y_true, y_pred):
    healthy = y_true == 0
    return {"macro_f1": f1_score(y_true, y_pred, labels=list(FAULTS), average="macro", zero_division=0),
            "worst_recall": recall_score(y_true, y_pred, labels=list(FAULTS), average=None, zero_division=0).min(),
            "false_alarm": float((y_pred[healthy] != 0).mean()),
            "accuracy": float((y_pred == y_true).mean())}


def mean(runs, group):
    return {**{k: float(np.mean([r[k] for r in runs])) for k in KEYS},
            "by": {g: {k: float(np.mean([r[k] for r in runs if r[group] == g])) for k in KEYS}
                   for g in dict.fromkeys(r[group] for r in runs)}}


def evaluate(stage="val"):
    """val: fit on train, score validation. test: fit on train+val, score the locked test. Means over SEEDS."""
    from src.experiment import fit_predict  # imported here so a broken experiment fails loudly at run time

    fit_part, eval_part = (0, 1) if stage == "val" else (1, 2)
    d = {}
    for m in MODES:
        x, y, files = data(m)
        d[m] = x, y, files, split(y, files)
    unseen, unseen_all, same = [], [], []
    for b in MODES:
        x, y, files, part = d[b]
        xa, ya, fa, pa = d[next(m for m in MODES if m != b)]
        fit, ev = part <= fit_part, part == eval_part
        half = second_half(y, files)
        commissioning = fit & (half == 0)  # each new plant's early healthy history, nothing later
        late = half[ev] == 1  # time-matched: every class's windows from the same stretch of its run
        offset = fa.max() + 1  # the other mode's plants are different plants
        x_u = np.concatenate([xa[pa <= fit_part], x[commissioning]])
        y_u = np.concatenate([ya[pa <= fit_part], y[commissioning]])
        f_u = np.concatenate([fa[pa <= fit_part], files[commissioning] + offset])
        for seed in SEEDS:
            pred = np.asarray(fit_predict(x_u, y_u, f_u, x[ev], files[ev] + offset, seed))
            unseen.append({"to": b, **score(y[ev][late], pred[late])})
            unseen_all.append({"to": b, **score(y[ev], pred)})
            pred = np.asarray(fit_predict(x[fit], y[fit], files[fit], x[ev], files[ev], seed))
            same.append({"to": b, **score(y[ev][late], pred[late])})
    return {"unseen": mean(unseen, "to"), "unseen_all_windows": mean(unseen_all, "to"), "same_plant": mean(same, "to")}


def keep(new, best):
    """The gate: a real gain, and neither the weakest class nor false alarms get worse."""
    return best is None or (new["macro_f1"] >= best["macro_f1"] + MIN_GAIN
                            and new["worst_recall"] >= best["worst_recall"]
                            and new["false_alarm"] <= best["false_alarm"])


def best_so_far():
    if not LOG.exists():
        return None, 0
    rows = list(csv.DictReader(LOG.open(encoding="utf-8"), delimiter="	"))
    kept = [r for r in rows if r["decision"] == "keep"]
    return ({k: float(kept[-1]["unseen_" + k]) for k in KEYS} if kept else None), len(rows)


def main():
    description = " ".join(sys.argv[1:]) or "no description"
    best, n = best_so_far()
    r = evaluate("val")
    decision = "keep" if keep(r["unseen"], best) else "discard"
    new_file = not LOG.exists()
    with LOG.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="	")
        if new_file:
            w.writerow(FIELDS)
        a = r["unseen_all_windows"]
        w.writerow([n, description, *(f"{r['unseen'][k]:.4f}" for k in KEYS), f"{a['macro_f1']:.4f}",
                    f"{a['false_alarm']:.4f}", f"{r['same_plant']['macro_f1']:.4f}", decision])
    print(json.dumps({**r, "best_before": best, "decision": decision.upper()}, indent=2))


if __name__ == "__main__":
    main()
