"""Detect-then-name: the loop-3 discard reported in the README (late false alarms 87% -> 26%, macro-F1 0.46 -> 0.39).
Reproduce: copy this file over src/experiment.py, then run  python -m src.final_test "detect-then-name".

Contract: fit_predict(x_fit, y_fit, files_fit, x_eval, files_eval, seed) -> predicted labels for x_eval.
x is raw windows (n, 13 signals, 1000 samples) in time order within each file. files_eval may name plants whose only
fit windows are early healthy ones (commissioning data): the model must name faults it has never seen on that plant.

This version splits the two questions. Detect: is the window beyond anything healthy operation showed, drift included?
Name: if so, which fault does the published classifier find most likely (healthy is not an option at that point)?
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import featurize


def site_z(F, files, ref_F, ref_files):
    """Express features relative to each experiment's own healthy fit-time windows."""
    out = np.empty_like(F)
    for f in np.unique(files):
        ref = ref_F[ref_files == f]
        out[files == f] = (F[files == f] - ref.mean(axis=0)) / (ref.std(axis=0) + 1e-6)
    return out


def drift_rows(F, y, files):
    """Healthy-only files: later half against the earlier half. How far healthy operation drifts with time."""
    rows = [np.empty((0, F.shape[1]))]
    for f in np.unique(files):
        Ff = F[files == f]
        if not y[files == f].any() and len(Ff) >= 4:
            early = Ff[: len(Ff) // 2]
            rows.append((Ff[len(Ff) // 2:] - early.mean(axis=0)) / (early.std(axis=0) + 1e-6))
    return np.concatenate(rows)


def fit_predict(x_fit, y_fit, files_fit, x_eval, files_eval, seed):
    F_fit, F_eval = featurize(x_fit.astype(np.float64)), featurize(x_eval.astype(np.float64))
    healthy = y_fit == 0
    Z_fit = site_z(F_fit, files_fit, F_fit[healthy], files_fit[healthy])
    Z_eval = site_z(F_eval, files_eval, F_fit[healthy], files_fit[healthy])
    # 1) detect: beyond the largest |z| healthy operation ever showed on that feature, drift included
    limit = np.abs(np.concatenate([Z_fit[healthy], drift_rows(F_fit, y_fit, files_fit)])).max(axis=0) + 1e-6
    flagged = (np.abs(Z_eval) / limit).max(axis=1) > 1
    # 2) name: the published classifier's most likely fault class
    clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed).fit(Z_fit, y_fit)
    pred = np.zeros(len(Z_eval), int)
    if flagged.any():
        proba = clf.predict_proba(Z_eval[flagged])
        proba[:, clf.classes_ == 0] = -1
        pred[flagged] = clf.classes_[proba.argmax(axis=1)]
    return pred
