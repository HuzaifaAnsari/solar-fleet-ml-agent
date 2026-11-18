"""Checks that need no dataset: split, baseline, detector and work-order logic on synthetic data."""
import numpy as np

from src.agent import PLAYBOOK, RANK, work_order
from src.features import SIGNALS, featurize
from src.prepare import keep, second_half, split
from src.train import control_chart, site_baseline, time_split


def test_time_split_keeps_order_and_ratio():
    labels = np.array([0] * 10 + [3] * 10)
    files = np.zeros(20, int)
    train = time_split(labels, files)
    assert train[:7].all() and not train[7:10].any()      # earlier 70% of each segment trains
    assert train[10:17].all() and not train[17:].any()


def test_site_baseline_uses_only_healthy_training_rows():
    F = np.array([[1.0], [3.0], [100.0], [500.0]])
    labels, files = np.array([0, 0, 0, 5]), np.zeros(4, int)
    train = np.array([True, True, False, False])           # row 2 is healthy but held out
    Z = site_baseline(F, labels, files, train)
    assert np.isclose(Z[0, 0], -1) and np.isclose(Z[1, 0], 1)  # mean 2, std 1 from rows 0-1 only


def test_control_chart_counts_precision_and_recall():
    Z = np.array([[0.1], [5.0], [9.0], [0.2]])
    y, test = np.array([0, 4, 1, 0]), np.ones(4, bool)
    r = control_chart(Z, y, test)["k=4"]
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["false_alarm_rate"] == 0.0


def test_featurize_energy_balance():
    x = np.ones((2, len(SIGNALS), 100))
    F = featurize(x)
    assert F.shape == (2, 2 * len(SIGNALS) + 5)
    assert np.allclose(F[:, 2 * len(SIGNALS) + 2], 3.0)   # 3 phases x 1 W out / 1 W DC in


def test_work_order_is_held_for_approval_and_ranked():
    r = {"plant": "P1", "fault": "inverter fault (IGBT)", "fault_id": 1, "confidence": 0.9,
         "started_s": 7.0, "power_lost_w": 200.0, "kwh_at_risk_per_day": 1.0}
    o = work_order(r)
    assert o["status"] == "awaiting approval" and o["severity"] == "critical"
    assert all(sev in RANK for sev, _, _ in PLAYBOOK.values())


def test_research_split_is_time_ordered_55_15_30_in_each_half_of_every_run():
    half = np.array([0] * 11 + [1] * 3 + [2] * 6)
    fault_run = split(np.r_[np.zeros(20, int), np.full(20, 3)], np.zeros(40, int))
    healthy_run = split(np.zeros(40, int), np.ones(40, int))  # its second half is tested like a fault's: drift shows up
    assert (fault_run == np.r_[half, half]).all() and (healthy_run == np.r_[half, half]).all()
    assert (second_half(np.zeros(40, int), np.ones(40, int)) == np.r_[np.zeros(20), np.ones(20)]).all()


def test_keep_gate_needs_real_gain_without_worse_recall_or_false_alarms():
    best = {"macro_f1": 0.5, "worst_recall": 0.1, "false_alarm": 0.05}
    assert keep(best, None) and keep({**best, "macro_f1": 0.52}, best)
    assert not keep({**best, "macro_f1": 0.505}, best)                         # within noise
    assert not keep({**best, "macro_f1": 0.6, "false_alarm": 0.06}, best)      # bought with false alarms
    assert not keep({**best, "macro_f1": 0.6, "worst_recall": 0.0}, best)      # a class got dropped
