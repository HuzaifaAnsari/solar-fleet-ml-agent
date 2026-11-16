"""Load GPVS-Faults experiments and cut them into labelled windows.

Each .mat file is one ~14 s experiment sampled at ~10 kHz. The dataset ReadMe states faults were
"introduced manually halfway", so the first half is healthy (label 0) and the second half is the
file's fault (label x of Fx). The exact onset isn't published, so a band around the midpoint is dropped.
"""
from pathlib import Path

import h5py
import numpy as np

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
COLUMNS = ["Time", "Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc", "Iabc", "If", "Vabc", "Vf"]
SIGNALS = COLUMNS[1:]
WINDOW = 1000          # samples, ~0.1 s
ONSET_BAND = (0.45, 0.55)  # fraction of the experiment dropped around the (unpublished) fault onset
FAULTS = {
    0: "healthy",
    1: "inverter fault (IGBT)",
    2: "feedback sensor fault",
    3: "grid anomaly",
    4: "PV array partial shading",
    5: "PV array open circuit",
    6: "boost converter controller fault (PI gain)",
    7: "boost converter controller fault",
}


def load(name: str) -> dict[str, np.ndarray]:
    """Read the MATLAB table: its 14 columns are stored as #refs# datasets in column order."""
    with h5py.File(RAW / f"{name}.mat", "r") as f:
        names = ["".join(map(chr, f[r][:].ravel())) for r in f["#refs#/v"][:].ravel()]
        cols = [f[r][:].ravel() for r in f["#refs#/c"][:].ravel()]
    assert names == COLUMNS, f"unexpected columns in {name}: {names}"
    return dict(zip(names, cols))


def windows(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw windows (n, len(SIGNALS), WINDOW), labels (n,), and window start times (n,)."""
    data = load(name)
    fault = int(name[1])
    x = np.stack([data[s] for s in SIGNALS])
    n = x.shape[1] // WINDOW
    x = x[:, : n * WINDOW].reshape(len(SIGNALS), n, WINDOW).transpose(1, 0, 2)
    frac = (np.arange(n) + 0.5) / n
    keep = (frac < ONSET_BAND[0]) | (frac > ONSET_BAND[1])
    labels = np.where(frac > ONSET_BAND[1], fault, 0)
    starts = data["Time"][: n * WINDOW : WINDOW]
    return x[keep], labels[keep], starts[keep]


def featurize(x: np.ndarray) -> np.ndarray:
    """Per-window statistics plus physics features the raw signals hide."""
    i = {s: k for k, s in enumerate(SIGNALS)}
    mean, std = x.mean(axis=2), x.std(axis=2)
    p_dc = (x[:, i["Vpv"]] * x[:, i["Ipv"]]).mean(axis=1)
    p_ac = sum((x[:, i[v]] * x[:, i[c]]).mean(axis=1) for v, c in (("va", "ia"), ("vb", "ib"), ("vc", "ic")))
    efficiency = p_ac / np.maximum(p_dc, 1e-6)          # energy balance across the inverter
    rms = np.sqrt((x[:, [i["ia"], i["ib"], i["ic"]]] ** 2).mean(axis=2))
    imbalance = rms.std(axis=1) / np.maximum(rms.mean(axis=1), 1e-6)  # phase current imbalance
    ripple = std[:, i["Vdc"]] / np.maximum(np.abs(mean[:, i["Vdc"]]), 1e-6)
    return np.column_stack([mean, std, p_dc, p_ac, efficiency, imbalance, ripple])


def dataset(mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All 8 experiments of one mode ('M' = MPPT, 'L' = IPPT): raw windows, labels, file ids, times."""
    xs, ys, files, times = [], [], [], []
    for fault in range(8):
        x, y, t = windows(f"F{fault}{mode}")
        xs.append(x), ys.append(y), files.append(np.full(len(y), fault)), times.append(t)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(files), np.concatenate(times)
