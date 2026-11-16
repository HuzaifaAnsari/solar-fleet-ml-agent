"""Download GPVS-Faults (Bakdi et al., Mendeley Data, CC BY 4.0, DOI 10.17632/n76t439f65.1) into data/raw/.

Not committed to git: run this once. Files: F{0..7}{L,M}.mat (fault 0 = healthy; L = IPPT, M = MPPT) + ReadMe.pdf.
"""
import json
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
API = "https://data.mendeley.com/public-api/datasets/n76t439f65/files?folder_id=root&version=1"
UA = {"User-Agent": "Mozilla/5.0 (solar-fleet-ml-agent data fetch)"}  # Mendeley 403s the default urllib agent


def get(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    files = json.load(get(API))
    for f in files:
        name = f["filename"]
        if not (name.endswith(".mat") or name == "ReadMe.pdf"):
            continue  # skip the 120 MB CSV zip: same data as the .mat files
        dest = RAW / name
        if dest.exists() and dest.stat().st_size == f["size"]:
            continue
        print("downloading", name)
        dest.write_bytes(get(f["content_details"]["download_url"]).read())
        assert dest.stat().st_size == f["size"], f"size mismatch for {name}"
    print("ok:", sorted(p.name for p in RAW.iterdir()))


if __name__ == "__main__":
    main()
