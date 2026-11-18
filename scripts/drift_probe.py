"""Does a healthy run drift, and can anything logged explain it? Healthy runs only (file 0 of each mode).

(1) Is time-in-run decodable? Train early-vs-late on one mode's healthy run, test on the other's.
(2) Does the drift follow a covariate faults don't drive? R^2 of each drifting feature regressed on grid-side signals.
(3) As a new plant would do it: fit that regression on the early (commissioning) quarter only, remove it, and probe again.

Run: python -m scripts.drift_probe   (needs the data from scripts/fetch_data.py)
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score

from src.features import SIGNALS, featurize
from src.prepare import data

NAMES = [f"mean_{s}" for s in SIGNALS] + [f"std_{s}" for s in SIGNALS] + ["p_dc", "p_ac", "eff", "imb", "ripple"]
GRID = [NAMES.index(k) for k in ("mean_Vabc", "std_Vabc", "mean_Vf", "std_Vf", "std_va", "std_vb", "std_vc")]
DRIFTING = ("mean_Vdc", "mean_ia", "imb", "mean_Vpv")
VDC = NAMES.index("mean_Vdc")
PAIRS = (("M", "L"), ("L", "M"))


def healthy_run(mode):
    """Features of the healthy run, z-scored against its first quarter; and 1 for windows in its second half."""
    x, _, files = data(mode)
    F = featurize(x[files == 0].astype(float))
    n, ref = len(F), F[: len(F) // 4]
    return (F - ref.mean(0)) / (ref.std(0) + 1e-6), (np.arange(n) >= n // 2).astype(int)


def time_probe(Z, late, a, b):
    """Balanced accuracy of telling early from late windows in mode b, trained on mode a (0.50 = not decodable)."""
    pred = HistGradientBoostingClassifier(random_state=0).fit(Z[a], late[a]).predict(Z[b])
    return balanced_accuracy_score(late[b], pred)


def grid_residual(Z):
    """Each non-grid feature minus its linear fit on grid-side signals, fitted on the first quarter only."""
    early, res = slice(0, len(Z) // 4), Z.copy()
    for j in range(Z.shape[1]):
        if j not in GRID:
            lr = LinearRegression().fit(Z[early][:, GRID], Z[early][:, j])
            res[:, j] = Z[:, j] - lr.predict(Z[:, GRID])
    return res


def main():
    Z, late = {}, {}
    for m in "ML":
        Z[m], late[m] = healthy_run(m)
    for a, b in PAIRS:
        print(f"(1) time-in-run probe {a}->{b}: balanced accuracy {time_probe(Z, late, a, b):.2f} (0.50 = not decodable)")
    for m in "ML":
        for k in DRIFTING:
            j = NAMES.index(k)
            r2 = LinearRegression().fit(Z[m][:, GRID], Z[m][:, j]).score(Z[m][:, GRID], Z[m][:, j])
            print(f"(2) {m} healthy run: {k} explained by grid-side signals R^2={r2:.2f}; "
                  f"drift late-vs-early {Z[m][late[m] == 1, j].mean():+.1f} sd")
    R = {m: grid_residual(Z[m]) for m in "ML"}
    for m in "ML":
        print(f"(3) {m}: mean_Vdc late drift raw {Z[m][late[m] == 1, VDC].mean():+.1f} sd "
              f"-> residual {R[m][late[m] == 1, VDC].mean():+.1f} sd")
    for a, b in PAIRS:
        print(f"(3) time-in-run probe on residuals {a}->{b}: balanced accuracy {time_probe(R, late, a, b):.2f}")


if __name__ == "__main__":
    main()
