"""The ONE file the research agent edits (see program.md). Baseline = the published model in src/train.py:
physics features -> per-site baseline z-scores -> balanced gradient boosting.

Contract: fit_predict(x_fit, y_fit, files_fit, x_eval, files_eval, seed) -> predicted labels for x_eval.
x is raw windows (n, 13 signals, 1000 samples) in time order within each file. files_eval may name plants whose only
fit windows are early healthy ones (commissioning data): the model must name faults it has never seen on that plant.

Loop 3 (time-matched gate) kept nothing, so this is still the published model. See reports/experiments.tsv.
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


def fit_predict(x_fit, y_fit, files_fit, x_eval, files_eval, seed):
    F_fit, F_eval = featurize(x_fit.astype(np.float64)), featurize(x_eval.astype(np.float64))
    healthy = y_fit == 0
    Z_fit = site_z(F_fit, files_fit, F_fit[healthy], files_fit[healthy])
    Z_eval = site_z(F_eval, files_eval, F_fit[healthy], files_fit[healthy])
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed).fit(Z_fit, y_fit).predict(Z_eval)
