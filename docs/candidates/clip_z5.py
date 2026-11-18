"""Loop-2 winner: the published model with site z-scores clipped at +-5 (README locked-test table, macro-F1 0.43).
Reproduce: copy this file over src/experiment.py, then run  python -m src.final_test "loop-2 winner (site z clipped at +-5)".

Same contract and features as src/experiment.py. The only change is np.clip on both z-score arrays: keep the direction
of a deviation, drop its mode-specific size. It won loop 2 on the all-window score and lost on the time-matched one.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import featurize

CLIP = 5.0


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
    Z_fit = np.clip(site_z(F_fit, files_fit, F_fit[healthy], files_fit[healthy]), -CLIP, CLIP)
    Z_eval = np.clip(site_z(F_eval, files_eval, F_fit[healthy], files_fit[healthy]), -CLIP, CLIP)
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed).fit(Z_fit, y_fit).predict(Z_eval)
