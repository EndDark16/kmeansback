"""
Helper script that regenerates the synthetic neighborhoods dataset.

It is not required at runtime but documents how the CSV in data/ was created.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "data" / "neighborhoods_synthetic.csv"

ZONES = {
    "downtown": {
        "population": (8000, 15000),
        "income": (900, 1600),
        "elderly": (0.08, 0.2),
        "chronic": (0.08, 0.18),
        "accidents": (45, 80),
        "road": (0.2, 2.0),
    },
    "low_income": {
        "population": (6000, 11000),
        "income": (350, 750),
        "elderly": (0.1, 0.28),
        "chronic": (0.15, 0.35),
        "accidents": (30, 65),
        "road": (0.5, 3.5),
    },
    "suburban": {
        "population": (3500, 7000),
        "income": (800, 1500),
        "elderly": (0.05, 0.18),
        "chronic": (0.05, 0.2),
        "accidents": (10, 35),
        "road": (1.0, 6.0),
    },
    "rural": {
        "population": (800, 4000),
        "income": (300, 900),
        "elderly": (0.12, 0.4),
        "chronic": (0.15, 0.45),
        "accidents": (5, 20),
        "road": (3.0, 10.0),
    },
}


def build_rows(total: int = 150) -> list[dict[str, float | int | str]]:
    random.seed(42)
    rows: list[dict[str, float | int | str]] = []
    for idx in range(1, total + 1):
        zone = random.choice(list(ZONES))
        spec = ZONES[zone]
        x = round(random.uniform(0, 20), 2)
        y = round(random.uniform(0, 20), 2)
        population = int(random.uniform(*spec["population"]))
        income = round(random.uniform(*spec["income"]), 2)
        elderly = round(random.uniform(*spec["elderly"]), 3)
        chronic = round(random.uniform(*spec["chronic"]), 3)
        accidents = int(random.uniform(*spec["accidents"]))
        road_distance = round(random.uniform(*spec["road"]), 2)

        demand = (
            0.35 * (population / 15000)
            + 0.25 * chronic
            + 0.15 * elderly
            + 0.15 * (accidents / 80)
            + 0.1 * (1 - min(road_distance / 10, 1))
        ) * 100
        demand_index = round(demand, 2)

        rows.append(
            {
                "id": idx,
                "zone": zone,
                "x_km": x,
                "y_km": y,
                "population": population,
                "median_income": income,
                "elderly_pct": elderly,
                "chronic_disease_pct": chronic,
                "accidents_per_year": accidents,
                "distance_to_main_road_km": road_distance,
                "demand_index": demand_index,
            }
        )
    return rows


def main() -> None:
    rows = build_rows()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "id",
                "zone",
                "x_km",
                "y_km",
                "population",
                "median_income",
                "elderly_pct",
                "chronic_disease_pct",
                "accidents_per_year",
                "distance_to_main_road_km",
                "demand_index",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Dataset written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
