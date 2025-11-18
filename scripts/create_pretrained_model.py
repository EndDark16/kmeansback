"""
Offline helper script that fits K-Means on the synthetic dataset to produce
pretrained centroids consumed by the API's /kmeans/pretrained endpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from random import Random

import csv

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "neighborhoods_synthetic.csv"
OUTPUT_PATH = BASE_DIR / "models" / "pretrained_hospitals.json"

sys.path.insert(0, str(BASE_DIR))

from app.kmeans import run_kmeans  # noqa: E402


def load_points() -> list[tuple[float, float]]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [(float(row["x_km"]), float(row["y_km"])) for row in reader]


def main(k: int = 4) -> None:
    points = load_points()
    result = run_kmeans(points, k=k, rng=Random(1234), max_iter=200, tol=1e-4)
    payload = {
        "k": k,
        "hospitals": [[round(x, 3), round(y, 3)] for x, y in result.centroids],
        "description": (
            f"Deterministic K-Means (k={k}) entrenado sobre {DATA_PATH.name} "
            "con coordenadas (x_km, y_km)."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Pretrained centroids saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
