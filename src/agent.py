"""Fleet agent: models detect and classify, the agent turns detections into ranked work orders for a human.

Each GPVS-Faults experiment is replayed as one "plant". Per plant:
  1. site baseline   - the plant's first healthy windows define its normal (no labels used at run time)
  2. detect          - control chart: any feature > k standard deviations from that baseline (decides IF)
  3. classify        - gradient-boosting classifier names the fault type, with a probability (decides WHAT)
  4. debounce        - an incident opens only after MIN_RUN consecutive abnormal windows
  5. energy at risk  - measured DC power drop vs baseline, times assumed daily sun hours
  6. work order      - playbook steps, priority by criticality and kWh at risk; status = awaiting approval
If OPENAI_API_KEY is set an LLM rewrites the work order in plain language; on failure the template stands.

Run: python -m src.agent            (replays the 8 MPPT experiments, writes reports/work_orders.md)
Note: this replays the experiments the classifier was trained on. Held-out accuracy is in reports/results.md.
"""
import json
import os
from pathlib import Path

import joblib  # loads only models/ produced locally by src/train.py, never downloaded files
import numpy as np

from src.features import FAULTS, SIGNALS, featurize, windows

ROOT = Path(__file__).resolve().parent.parent
WARMUP = 0.3     # fraction of each replay used as the site's healthy baseline
MIN_RUN = 3      # consecutive abnormal windows before an incident opens (~0.3 s)
SUN_HOURS = 5.0  # assumption: equivalent full-sun hours per day, used to turn watts lost into kWh/day
PLAYBOOK = {
    1: ("critical", "Isolate the inverter, check IGBT gate drivers and DC-link capacitors, replace the faulty module.", "electrician + inverter OEM"),
    2: ("high", "Compare the phase-voltage sensor against a calibrated meter; recalibrate or replace the sensor.", "instrumentation tech"),
    3: ("high", "Log grid voltage at the point of connection; raise with the utility if fluctuations persist.", "grid liaison"),
    4: ("medium", "Inspect the array for shading or soiling, clean or trim, check bypass diodes on affected strings.", "O&M crew"),
    5: ("high", "Locate the open string with string-level current checks; repair connectors or cabling.", "O&M crew"),
    6: ("medium", "Review boost-converter PI controller settings against commissioning values; restore and retest.", "controls engineer"),
    7: ("medium", "Run controller diagnostics on the boost converter; reflash firmware or replace the control board.", "controls engineer"),
}
RANK = {"critical": 0, "high": 1, "medium": 2}


def plant_report(name: str, gbm, k: float) -> dict:
    x, _, times = windows(name)  # labels are ignored at run time
    F = featurize(x)
    i_vpv, i_ipv = SIGNALS.index("Vpv"), SIGNALS.index("Ipv")
    p_dc = (x[:, i_vpv] * x[:, i_ipv]).mean(axis=1)
    n0 = max(int(len(F) * WARMUP), 5)
    mu, sd = F[:n0].mean(axis=0), F[:n0].std(axis=0) + 1e-6
    Z = (F - mu) / sd
    abnormal = np.abs(Z).max(axis=1) > k
    proba = gbm.predict_proba(Z)
    pred = gbm.classes_[proba.argmax(1)]
    # The control chart decides *whether* there is an incident; the classifier only names it. The classifier
    # alone over-calls "grid anomaly" on slow healthy drift (see README limitations), so it can't open one.

    incidents, run = [], []
    for t in range(n0, len(F)):
        if abnormal[t]:
            run.append(t)
            continue
        if len(run) >= MIN_RUN:
            incidents.append(run)
        run = []
    if len(run) >= MIN_RUN:
        incidents.append(run)
    if not incidents:
        return {"plant": name, "status": "healthy"}

    idx = max(incidents, key=len)  # the longest abnormal stretch
    faulty = [t for t in idx if pred[t] != 0] or idx
    cls = int(np.bincount(pred[faulty].astype(int)).argmax()) or 0
    lost_w = max(float(p_dc[:n0].mean() - p_dc[idx].mean()), 0.0)
    return {
        "plant": name, "status": "incident", "fault": FAULTS.get(cls, "unclassified anomaly"), "fault_id": cls,
        "confidence": round(float(proba[faulty, list(gbm.classes_).index(cls)].mean()), 2) if cls else None,
        "started_s": round(float(times[idx[0]]), 2), "windows": len(idx),
        "power_lost_w": round(lost_w, 1), "kwh_at_risk_per_day": round(lost_w * SUN_HOURS / 1000, 3),
    }


def work_order(r: dict) -> dict:
    severity, action, crew = PLAYBOOK.get(r["fault_id"], ("medium", "Investigate the abnormal signals on site.", "O&M crew"))
    text = (f"{r['plant']}: suspected {r['fault']} (model confidence {r['confidence']}), detected at t={r['started_s']}s. "
            f"Estimated loss {r['power_lost_w']} W, about {r['kwh_at_risk_per_day']} kWh/day if left unrepaired. "
            f"Action: {action} Assign: {crew}.")
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            text = OpenAI().chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0,
                messages=[{"role": "system", "content": "Rewrite this maintenance work order for a field technician in plain language. "
                           "Keep every number and the recommended action exactly; add nothing that isn't in it."},
                          {"role": "user", "content": text}],
            ).choices[0].message.content.strip()
        except Exception as exc:  # the template is the fallback when the model fails
            text += f" [LLM unavailable: {type(exc).__name__}]"
    return {**r, "severity": severity, "work_order": text, "status": "awaiting approval"}


def run_fleet(mode: str = "M") -> list[dict]:
    models = joblib.load(ROOT / "models" / "sklearn.joblib")
    reports = [plant_report(f"F{f}{mode}", models["gbm"], models["chart_k"]) for f in range(8)]
    orders = [work_order(r) for r in reports if r["status"] == "incident"]
    return sorted(orders, key=lambda o: (RANK[o["severity"]], -o["kwh_at_risk_per_day"])), [r for r in reports if r["status"] == "healthy"]


def main():
    orders, healthy = run_fleet()
    lines = ["# Work-order queue (awaiting human approval)", "",
             "| # | Plant | Suspected fault | Confidence | Severity | Power lost | kWh/day at risk |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {i} | {o['plant']} | {o['fault']} | {o['confidence']} | {o['severity']} | {o['power_lost_w']} W | {o['kwh_at_risk_per_day']} |"
              for i, o in enumerate(orders, 1)]
    lines += ["", f"Healthy plants (no incident): {', '.join(h['plant'] for h in healthy) or 'none'}", ""]
    lines += [f"**{o['plant']}**: {o['work_order']}" + "\n" for o in orders]
    (ROOT / "reports" / "work_orders.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "reports" / "work_orders.json").write_text(json.dumps(orders, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
