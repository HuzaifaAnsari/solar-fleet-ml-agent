"""Healthy runs only. (1) Is time-in-run decodable? Train early-vs-late on one mode's healthy run, test on the other's.
(2) Does the drift follow a covariate faults don't drive? R^2 of each drifting feature regressed on grid-side signals."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score
from src.features import featurize, SIGNALS
from src.prepare import data
names = ["mean_"+s for s in SIGNALS]+["std_"+s for s in SIGNALS]+["p_dc","p_ac","eff","imb","ripple"]
Z, T = {}, {}
for m in "ML":
    x, y, f = data(m); h = f == 0
    F = featurize(x[h].astype(float)); n = len(F); ref = F[: n // 4]
    Z[m] = (F - ref.mean(0)) / (ref.std(0) + 1e-6); T[m] = (np.arange(n) >= n // 2).astype(int)
for a, b in (("M", "L"), ("L", "M")):
    p = HistGradientBoostingClassifier(random_state=0).fit(Z[a], T[a]).predict(Z[b])
    print(f"(1) time-in-run probe {a}->{b}: balanced accuracy {balanced_accuracy_score(T[b], p):.2f} (0.50 = not decodable)")
grid = [names.index(k) for k in ("mean_Vabc", "std_Vabc", "mean_Vf", "std_Vf", "std_va", "std_vb", "std_vc")]
for m in "ML":
    for k in ("mean_Vdc", "mean_ia", "imb", "mean_Vpv"):
        j = names.index(k); r2 = LinearRegression().fit(Z[m][:, grid], Z[m][:, j]).score(Z[m][:, grid], Z[m][:, j])
        print(f"(2) {m} healthy run: {k} explained by grid-side signals R^2={r2:.2f}; drift late-vs-early {Z[m][T[m]==1, j].mean():+.1f} sd")
# (3) Realistic: residual model fitted on the EARLY (commissioning) quarter only, then applied to late windows.
R = {}
for m in "ML":
    n = len(Z[m]); e = slice(0, n // 4); res = Z[m].copy()
    for j in range(Z[m].shape[1]):
        if j in grid: continue
        lr = LinearRegression().fit(Z[m][e][:, grid], Z[m][e][:, j]); res[:, j] = Z[m][:, j] - lr.predict(Z[m][:, grid])
    R[m] = res
    j = names.index("mean_Vdc")
    print(f"(3) {m}: mean_Vdc late drift raw {Z[m][T[m]==1, j].mean():+.1f} sd -> residual {R[m][T[m]==1, j].mean():+.1f} sd")
for a, b in (("M", "L"), ("L", "M")):
    p = HistGradientBoostingClassifier(random_state=0).fit(R[a], T[a]).predict(R[b])
    print(f"(3) time-in-run probe on residuals {a}->{b}: balanced accuracy {balanced_accuracy_score(T[b], p):.2f}")
