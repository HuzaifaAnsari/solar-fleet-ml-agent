# Research program (instructions for the agent running the loop)

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch): one fixed harness, one editable file, keep or discard.

## Goal
Raise **time-matched, unseen-plant** validation macro-F1 of the 8-class PV fault classifier: fit on one control mode's
8 experiments plus the other mode's commissioning windows (first halves of its runs), then name faults on the other
mode's experiments, scoring second-half windows only. Averaged over both directions (MPPT->IPPT, IPPT->MPPT) and 3 seeds.

Why unseen plants: every GPVS experiment has exactly one fault type, and a classifier names the experiment from
healthy windows alone with 100% accuracy. A first loop that optimised same-plant F1 went 0.80 -> 0.91 while unseen-plant
F1 fell 0.50 -> 0.19: it learned plant fingerprints, not faults (`reports/experiments_within_plant.tsv`).

Why time-matched: every run drifts over its 14 s, and fault windows are all late in a run. Scored on all windows, a
model can call "late" a fault: the second loop's winner called 78-94% of the healthy run's late windows a fault
(`reports/experiments_unseen_plant.tsv`, `reports/experiments_drift_check.tsv`). Second-half windows put every class
in the same stretch of time. All-window and same-plant scores are still logged; they are information, not the target.

## Files
- `src/prepare.py` **fixed.** Data, the time-ordered 55/15/30 split, scoring, the keep gate. Do not edit.
- `src/features.py` **fixed.** Windowing and labels. Do not edit.
- `src/experiment.py` **yours.** Edit freely; keep the `fit_predict(x_fit, y_fit, files_fit, x_eval, files_eval, seed)` contract.
- `src/final_test.py` **not yours.** Run by a human after the loop ends. Never run it, never read `reports/final_test.json` during the loop.

## One experiment
1. Make one idea's worth of change in `src/experiment.py`.
2. `python -m src.prepare "<one line: what changed and why>"`
3. It appends a row to `reports/experiments.tsv` and prints KEEP or DISCARD. The gate (`keep()` in prepare.py):
   time-matched unseen-plant macro-F1 up by at least 0.01, worst-class recall not lower, and the share of the
   healthy run's late windows called a fault not higher.
4. DISCARD: restore the last kept `src/experiment.py`. KEEP: it becomes the new base.

## Rules
- Use only installed packages (numpy, scikit-learn, torch). No new dependencies.
- A window's features may use that window and earlier windows only. No future data, no test labels.
- Fit everything (scalers, baselines, models) on the fit windows only; `x_eval` labels are never given to you.
- You may inspect validation predictions (confusion matrices, per-class recall) to choose the next idea.
- Stop after 40 experiments, or 10 in a row without a KEEP.
- Every attempt stays in `experiments.tsv`, including failures. Report "best of N tried".
