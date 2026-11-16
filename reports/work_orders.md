# Work-order queue (awaiting human approval)

| # | Plant | Suspected fault | Confidence | Severity | Power lost | kWh/day at risk |
|---|---|---|---|---|---|---|
| 1 | F1M | inverter fault (IGBT) | 1.0 | critical | 196.1 W | 0.98 |
| 2 | F5M | PV array open circuit | 0.94 | high | 188.5 W | 0.942 |
| 3 | F2M | feedback sensor fault | 1.0 | high | 0.1 W | 0.0 |
| 4 | F6M | boost converter controller fault (PI gain) | 1.0 | medium | 3.3 W | 0.016 |
| 5 | F7M | boost converter controller fault | 0.8 | medium | 0.0 W | 0.0 |

Healthy plants (no incident): F0M, F3M, F4M

**F1M**: F1M: suspected inverter fault (IGBT) (model confidence 1.0), detected at t=8.4s. Estimated loss 196.1 W, about 0.98 kWh/day if left unrepaired. Action: Isolate the inverter, check IGBT gate drivers and DC-link capacitors, replace the faulty module. Assign: electrician + inverter OEM.

**F5M**: F5M: suspected PV array open circuit (model confidence 0.94), detected at t=7.9s. Estimated loss 188.5 W, about 0.942 kWh/day if left unrepaired. Action: Locate the open string with string-level current checks; repair connectors or cabling. Assign: O&M crew.

**F2M**: F2M: suspected feedback sensor fault (model confidence 1.0), detected at t=8.7s. Estimated loss 0.1 W, about 0.0 kWh/day if left unrepaired. Action: Compare the phase-voltage sensor against a calibrated meter; recalibrate or replace the sensor. Assign: instrumentation tech.

**F6M**: F6M: suspected boost converter controller fault (PI gain) (model confidence 1.0), detected at t=12.2s. Estimated loss 3.3 W, about 0.016 kWh/day if left unrepaired. Action: Review boost-converter PI controller settings against commissioning values; restore and retest. Assign: controls engineer.

**F7M**: F7M: suspected boost converter controller fault (model confidence 0.8), detected at t=11.2s. Estimated loss 0.0 W, about 0.0 kWh/day if left unrepaired. Action: Run controller diagnostics on the boost converter; reflash firmware or replace the control board. Assign: controls engineer.
