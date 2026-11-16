"""Train and evaluate the three models; write reports/metrics.json, reports/results.md and models/.

Split: inside every experiment segment, the earlier 70% of windows train and the later 30% test.
Neighbouring windows are near-identical, so a random shuffle would leak answers into the test set.
Robustness: the classifiers are also trained on MPPT data and tested on IPPT data (and the reverse).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.features import FAULTS, dataset, featurize

ROOT = Path(__file__).resolve().parent.parent
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
TRAIN_FRAC = 0.7
POOL = 4  # 1000-sample windows downsampled to 250 for the CNN
SEED = 0


def time_split(labels, files):
    """Earlier 70% of each (experiment, label) segment trains; the later 30% tests."""
    train = np.zeros(len(labels), bool)
    for f in np.unique(files):
        for y in np.unique(labels[files == f]):
            idx = np.flatnonzero((files == f) & (labels == y))
            train[idx[: int(len(idx) * TRAIN_FRAC)]] = True
    return train


def site_baseline(F, labels, files, train):
    """Express each experiment's features relative to its own earlier healthy windows ("normal for this site").

    Uses only healthy *training* windows of the same experiment, which precede every test window in time.
    """
    out = np.empty_like(F)
    for f in np.unique(files):
        ref = F[(files == f) & (labels == 0) & train]
        out[files == f] = (F[files == f] - ref.mean(axis=0)) / (ref.std(axis=0) + 1e-6)
    return out


class CNN(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, classes),
        )

    def forward(self, x):
        return self.net(x)


def cnn_input(x, mean, std):
    x = x.reshape(*x.shape[:2], -1, POOL).mean(axis=3)
    return torch.tensor((x - mean) / std, dtype=torch.float32)


def train_cnn(x, y, epochs=40):
    torch.manual_seed(SEED)
    pooled = x.reshape(*x.shape[:2], -1, POOL).mean(axis=3)
    mean, std = pooled.mean(axis=(0, 2), keepdims=True), pooled.std(axis=(0, 2), keepdims=True) + 1e-6
    model = CNN(x.shape[1], len(FAULTS))
    weights = torch.tensor(len(y) / (len(FAULTS) * np.bincount(y, minlength=len(FAULTS)).clip(1)), dtype=torch.float32)
    opt, loss_fn = torch.optim.Adam(model.parameters(), 1e-3), nn.CrossEntropyLoss(weight=weights)
    xt, yt = cnn_input(x, mean, std), torch.tensor(y)
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(yt)).split(64):
            opt.zero_grad()
            loss_fn(model(xt[b]), yt[b]).backward()
            opt.step()
    model.eval()
    return model, mean, std


def cnn_predict(model, x, mean, std):
    with torch.no_grad():
        return model(cnn_input(x, mean, std)).argmax(1).numpy()


def scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(FAULTS), zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "per_class": {FAULTS[k]: {"precision": p[k], "recall": r[k], "f1": f[k]} for k in FAULTS}}


def detector(F_train, y_train, F_test, y_test):
    """Unsupervised: learn healthy behaviour only, flag anything unlike it."""
    scaler = StandardScaler().fit(F_train[y_train == 0])
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(scaler.transform(F_train[y_train == 0]))
    healthy_scores = -iso.score_samples(scaler.transform(F_train[y_train == 0]))
    threshold = np.quantile(healthy_scores, 0.99)  # ~1% false alarms on healthy training windows
    flagged = -iso.score_samples(scaler.transform(F_test)) > threshold
    fault = y_test > 0
    tp, fp = (flagged & fault).sum(), (flagged & ~fault).sum()
    return (scaler, iso, threshold), {
        "precision": tp / max(flagged.sum(), 1), "recall": tp / max(fault.sum(), 1),
        "false_alarm_rate": fp / max((~fault).sum(), 1),
        "recall_by_fault": {FAULTS[k]: float(flagged[y_test == k].mean()) for k in FAULTS if k},
    }


CHART_K = (4, 6, 8)  # false-alarm budgets reported side by side; k=4 is what the agent uses


def control_chart(Z, y, test):
    """Flag a window when any site-baseline feature is more than k standard deviations from normal."""
    s, fault = np.abs(Z[test]).max(axis=1), y[test] > 0
    out = {}
    for k in CHART_K:
        flag = s > k
        tp, fp = (flag & fault).sum(), (flag & ~fault).sum()
        out[f"k={k}"] = {"precision": tp / max(flag.sum(), 1), "recall": tp / max(fault.sum(), 1),
                         "false_alarm_rate": fp / max((~fault).sum(), 1),
                         "recall_by_fault": {FAULTS[c]: float(flag[y[test] == c].mean()) for c in FAULTS if c}}
    return out


def main():
    MODELS.mkdir(exist_ok=True), REPORTS.mkdir(exist_ok=True)
    data = {m: dataset(m) for m in "ML"}
    metrics = {"split": "time-ordered 70/30 inside each experiment segment", "modes": {}}

    for mode in "ML":
        x, y, files, _ = data[mode]
        tr = time_split(y, files)
        F_raw = featurize(x)
        F = site_baseline(F_raw, y, files, tr)
        _, det_raw = detector(F_raw[tr], y[tr], F_raw[~tr], y[~tr])
        chart = control_chart(F, y, ~tr)
        gbm_raw = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED).fit(F_raw[tr], y[tr])
        gbm = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED).fit(F[tr], y[tr])
        cnn, mu, sd = train_cnn(x[tr], y[tr])
        always_healthy = np.zeros((~tr).sum(), int)
        metrics["modes"][mode] = {
            "n_train": int(tr.sum()), "n_test": int((~tr).sum()),
            "detector_isolation_forest_raw": det_raw,
            "detector_control_chart_site_baseline": chart,
            "classifier_gbm_raw_features": scores(y[~tr], gbm_raw.predict(F_raw[~tr])),
            "classifier_gradient_boosting": scores(y[~tr], gbm.predict(F[~tr])),
            "classifier_1d_cnn": scores(y[~tr], cnn_predict(cnn, x[~tr], mu, sd)),
            "baseline_always_healthy": scores(y[~tr], always_healthy),
        }
        if mode == "M":  # ship the MPPT models for the agent demo
            joblib.dump({"gbm": gbm, "chart_k": CHART_K[0]}, MODELS / "sklearn.joblib")
            torch.save({"state": cnn.state_dict(), "mean": mu, "std": sd}, MODELS / "cnn.pt")

    # robustness: does a model trained in one operating mode survive the other?
    metrics["cross_mode"] = {}
    for a, b in (("M", "L"), ("L", "M")):
        (xa, ya, fa, _), (xb, yb, fb, _) = data[a], data[b]
        ta, tb = time_split(ya, fa), time_split(yb, fb)
        Fa, Fb = site_baseline(featurize(xa), ya, fa, ta), site_baseline(featurize(xb), yb, fb, tb)
        gbm = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED).fit(Fa[ta], ya[ta])
        metrics["cross_mode"][f"train_{a}_test_{b}"] = {"macro_f1": f1_score(yb[~tb], gbm.predict(Fb[~tb]), average="macro")}

    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
    write_results(metrics)


def write_results(m):
    pct = lambda v: f"{v:.0%}"
    lines = ["| Mode | Model | Accuracy | Macro-F1 | Worst-class recall |", "|---|---|---|---|---|"]
    names = {"baseline_always_healthy": "Always 'healthy' (baseline)", "classifier_gbm_raw_features": "Gradient boosting, raw features",
             "classifier_gradient_boosting": "Gradient boosting, site-baseline features",
             "classifier_1d_cnn": "1D-CNN on raw signals"}
    for mode, r in m["modes"].items():
        for key, label in names.items():
            s = r[key]
            worst = min(v["recall"] for v in s["per_class"].values())
            lines.append(f"| {'MPPT' if mode == 'M' else 'IPPT'} | {label} | {pct(s['accuracy'])} | {s['macro_f1']:.2f} | {pct(worst)} |")
    lines += ["", "| Mode | Unsupervised detector | Precision | Recall | False alarms on healthy |", "|---|---|---|---|---|"]
    for mode, r in m["modes"].items():
        md = "MPPT" if mode == "M" else "IPPT"
        d = r["detector_isolation_forest_raw"]
        lines.append(f"| {md} | Isolation Forest, raw features | {pct(d['precision'])} | {pct(d['recall'])} | {pct(d['false_alarm_rate'])} |")
        for k, d in r["detector_control_chart_site_baseline"].items():
            lines.append(f"| {md} | Control chart on site baseline, {k} | {pct(d['precision'])} | {pct(d['recall'])} | {pct(d['false_alarm_rate'])} |")
    lines += ["", "Cross-mode (train on one operating mode, test on the other), gradient boosting with site baselines, macro-F1: "
              + ", ".join(f"{k.replace('_', ' ')} = {v['macro_f1']:.2f}" for k, v in m["cross_mode"].items())]
    (REPORTS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
