"""Run after the loop ends, once per model you report: retrain src/experiment.py on train+validation, score the locked test.

Writes reports/final_test.json. Never run this inside the loop: choosing changes by test score would make it dishonest.
"""
import json
import sys

from src.prepare import ROOT, evaluate


def main():
    label = " ".join(sys.argv[1:]) or "final"
    r = evaluate("test")
    path = ROOT / "reports" / "final_test.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    results[label] = r
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
