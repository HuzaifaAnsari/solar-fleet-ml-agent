# Solar fleet ops: ML detects, an agent acts

An LLM agent is good at turning a detection into a clear work order. It is not a fault detector. This repo splits the job:

- **Physics features and a per-site baseline** say what "normal" looks like for each plant.
- **ML models** decide whether something is wrong and what it is, with numbers you can audit.
- **The agent** turns those detections into a ranked work-order queue that a human approves.

Everything runs on real, labelled fault experiments from a grid-connected PV system ([GPVS-Faults](https://data.mendeley.com/datasets/n76t439f65/1)), with no API key needed.

```
sensor windows ──► physics features ──► per-site baseline (z-scores)
                    (DC vs AC power,          │
                     phase imbalance,         ├─► control chart ──► IS something wrong?   (unsupervised)
                     DC-link ripple)          └─► gradient boosting ► WHAT is it?          (supervised, 8 classes)
                                                        │
                       debounce ► incident ► power lost vs baseline ► kWh/day at risk
                                                        │
                       playbook + priority ► work order ► awaiting human approval   (optional LLM rewrite, template fallback)
```

## Results (held-out data)

Split: inside each experiment, the earlier 70% of windows train and the later 30% test. Neighbouring 0.1 s windows are nearly identical, so a random shuffle would leak answers into the test set.

| Mode | Model | Accuracy | Macro-F1 | Worst-class recall |
|---|---|---|---|---|
| MPPT | Always "healthy" (baseline) | 56% | 0.09 | 0% |
| MPPT | Gradient boosting, raw features | 84% | 0.83 | 75% |
| MPPT | Gradient boosting, site-baseline features | 83% | 0.82 | 76% |
| MPPT | 1D-CNN on raw signals | 67% | 0.70 | 47% |
| IPPT | Always "healthy" (baseline) | 56% | 0.09 | 0% |
| IPPT | Gradient boosting, raw features | 83% | 0.74 | 0% |
| IPPT | Gradient boosting, site-baseline features | 81% | 0.74 | 0% |
| IPPT | 1D-CNN on raw signals | 42% | 0.51 | 0% |

**Why accuracy isn't the metric:** a model that never raises an alarm scores 56% accuracy here, with a macro-F1 of 0.09. On a real fleet, where faults are far rarer than in this dataset, the same do-nothing model would score well above 90%. Per-fault recall and the false-alarm rate are what an operator needs.

| Mode | Unsupervised detector | Precision | Recall | False alarms on healthy |
|---|---|---|---|---|
| MPPT | Isolation Forest, raw features | 100% | 28% | 0% |
| MPPT | Control chart on site baseline, k=4 | 96% | 66% | 2% |
| MPPT | Control chart on site baseline, k=6 | 100% | 54% | 0% |
| IPPT | Isolation Forest, raw features | 0% | 0% | 0% |
| IPPT | Control chart on site baseline, k=4 | 95% | 68% | 3% |
| IPPT | Control chart on site baseline, k=6 | 99% | 58% | 1% |

**Per-site baselines generalise.** Training on one operating mode and testing on the other, gradient-boosting macro-F1 rises from **0.18 to 0.49** (MPPT→IPPT) and from **0.31 to 0.52** (IPPT→MPPT) once features are expressed relative to each site's own healthy history.

**The two layers cover each other.** The control chart misses partial shading entirely (0% recall), but the supervised classifier catches it at 100% (MPPT) and 95% (IPPT) recall.

Full numbers: [`reports/metrics.json`](reports/metrics.json) and [`reports/results.md`](reports/results.md).

## The agent's output

`python -m src.agent` replays the eight MPPT experiments as eight plants and produces [`reports/work_orders.md`](reports/work_orders.md):

| # | Plant | Suspected fault | Confidence | Severity | Power lost | kWh/day at risk |
|---|---|---|---|---|---|---|
| 1 | F1M | inverter fault (IGBT) | 1.0 | critical | 196.1 W | 0.98 |
| 2 | F5M | PV array open circuit | 0.94 | high | 188.5 W | 0.942 |
| 3 | F2M | feedback sensor fault | 1.0 | high | 0.1 W | 0.0 |
| 4 | F6M | boost converter controller fault (PI gain) | 1.0 | medium | 3.3 W | 0.016 |
| 5 | F7M | boost converter controller fault | 0.8 | medium | 0.0 W | 0.0 |

The healthy plant (F0M) raises no incident. Every work order is held at `awaiting approval`, so a person decides.

## Limitations

- **The classifier over-calls "grid anomaly" on slow healthy drift.** Only 31 grid-anomaly windows exist in MPPT mode. So the agent lets the control chart decide *whether* there's an incident, and uses the classifier only to *name* it. The cost: the subtle grid-anomaly (F3M) and partial-shading (F4M) plants pass without an incident in the replay. More labelled data, or a rolling baseline, is the fix.
- **Fault onset isn't published.** The dataset says faults were "introduced manually halfway", so the middle 10% of each experiment is dropped rather than guessed.
- **k for the control chart** was compared after seeing these results; the table shows the trade-off across k instead of claiming one tuned value.
- **The agent replays the experiments the classifier trained on.** The held-out numbers above are the honest estimate.
- **This is lab data at ~200 W scale.** kWh/day assumes 5 full-sun hours (`SUN_HOURS` in `src/agent.py`).
- **The CNN underperforms gradient boosting.** With ~650 training windows per mode, hand-built physics features beat learned ones.

## Next

- A day-ahead forecast layer on real field data (NREL PVDAQ, CC BY 4.0), comparing a time-series foundation model against a seasonal-naive baseline, to price downtime properly.
- A rolling healthy baseline, so slow drift stops looking like a fault.

## Run it

```bash
pip install -r requirements.txt
python scripts/fetch_data.py   # ~85 MB from Mendeley Data
python -m src.train            # trains, evaluates, writes reports/ and models/
python -m src.agent            # replays the fleet, writes reports/work_orders.md
python -m pytest -q            # data-free unit tests (also run in CI)
```

Optional: set `OPENAI_API_KEY` (and `OPENAI_MODEL`) to have an LLM rewrite each work order in plain language. If the call fails, the template work order stands.

## Data and licence

Data: A. Bakdi, A. Guichi, S. Mekhilef, W. Bounoua, *GPVS-Faults: Experimental Data for fault scenarios in grid-connected PV systems under MPPT and IPPT modes*, Mendeley Data, V1, 2020, [doi:10.17632/n76t439f65.1](https://doi.org/10.17632/n76t439f65.1), CC BY 4.0. It is downloaded by the script, not redistributed here.

Code: MIT. Author: [Huzaifa Ansari](https://uzaifadev.com).
