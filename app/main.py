from __future__ import annotations

import json
import os
from pathlib import Path
from random import Random
from typing import List

import math
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .kmeans import KMeansResult, run_kmeans
from .schemas import (
    ClusterStats,
    DistanceBin,
    Hospital,
    KMeansRequest,
    KMeansResponse,
    Neighborhood,
    PretrainedModelResponse,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "pretrained_hospitals.json"


def get_cors_origins() -> List[str]:
    """
    Parse CORS origins from CORS_ORIGINS env var.

    Defaults to wildcard which is convenient for prototypes and Vercel previews.
    """
    raw = os.getenv("CORS_ORIGINS")
    if not raw:
        return [
            "http://localhost:5173",
            "https://kmeansfront.vercel.app",
        ]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_model_path() -> Path:
    raw = os.getenv("PRETRAINED_MODEL_PATH")
    if raw:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else BASE_DIR / candidate
    return DEFAULT_MODEL_PATH


app = FastAPI(title="Hospital Placement K-Means API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def simulate_neighborhoods(m: int, n: int, rng: Random) -> List[Neighborhood]:
    """Generate neighborhoods with random coordinates inside an m x m grid."""
    neighborhoods: List[Neighborhood] = []
    for idx in range(1, n + 1):
        x = rng.uniform(0, m)
        y = rng.uniform(0, m)
        neighborhoods.append(Neighborhood(id=idx, x=x, y=y))
    return neighborhoods


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/kmeans/run", response_model=KMeansResponse)
def run_kmeans_endpoint(payload: KMeansRequest) -> KMeansResponse:
    seed = payload.seed if payload.seed is not None else Random().randrange(1, 10_000_000)
    rng = Random(seed)
    neighborhoods = simulate_neighborhoods(payload.m, payload.n, rng)

    if payload.k > payload.n:
        raise HTTPException(
            status_code=400,
            detail="k cannot be greater than the number of simulated neighborhoods",
        )

    points = [(n.x, n.y) for n in neighborhoods]
    result: KMeansResult = run_kmeans(
        points,
        payload.k,
        max_iter=payload.max_iter,
        tol=payload.tol,
        rng=rng,
    )
    hospitals = [
        Hospital(id=idx + 1, x=centroid[0], y=centroid[1])
        for idx, centroid in enumerate(result.centroids)
    ]

    (
        cluster_stats,
        overall_avg_distance,
        overall_max_distance,
        inertia,
        overall_distances,
    ) = compute_cluster_stats(
        neighborhoods,
        hospitals,
        result.assignments,
    )
    distance_bins = build_distance_bins(overall_distances)
    return KMeansResponse(
        neighborhoods=neighborhoods,
        hospitals=hospitals,
        assignments=result.assignments,
        iterations=result.iterations,
        grid_size=payload.m,
        cluster_stats=cluster_stats,
        overall_avg_distance=overall_avg_distance,
        overall_max_distance=overall_max_distance,
        inertia=inertia,
        distance_bins=distance_bins,
    )


@app.get("/kmeans/pretrained", response_model=PretrainedModelResponse)
def get_pretrained_model() -> PretrainedModelResponse:
    model_path = get_model_path()
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Pretrained model not found at {model_path}")

    with model_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not {"k", "hospitals", "description"} <= payload.keys():
        raise HTTPException(status_code=500, detail="Invalid pretrained model format")

    return PretrainedModelResponse(**payload)


def compute_cluster_stats(
    neighborhoods: List[Neighborhood],
    hospitals: List[Hospital],
    assignments: List[int],
) -> tuple[List[ClusterStats], float, float, float, List[float]]:
    distances_by_cluster: dict[int, List[float]] = defaultdict(list)
    overall_distances: List[float] = []

    for neighborhood, cluster in zip(neighborhoods, assignments):
        hospital = hospitals[cluster]
        distance = math.hypot(neighborhood.x - hospital.x, neighborhood.y - hospital.y)
        distances_by_cluster[cluster].append(distance)
        overall_distances.append(distance)

    inertia = sum(distance**2 for distance in overall_distances)
    cluster_stats: List[ClusterStats] = []
    for idx, hospital in enumerate(hospitals):
        distances = distances_by_cluster.get(idx, [])
        count = len(distances)
        avg_distance = sum(distances) / count if count else 0.0
        max_distance = max(distances) if distances else 0.0
        cluster_stats.append(
            ClusterStats(
                hospital_id=hospital.id,
                count=count,
                avg_distance=avg_distance,
                max_distance=max_distance,
            )
        )

    overall_avg_distance = (
        sum(overall_distances) / len(overall_distances) if overall_distances else 0.0
    )
    overall_max_distance = max(overall_distances) if overall_distances else 0.0
    return cluster_stats, overall_avg_distance, overall_max_distance, inertia, overall_distances


def build_distance_bins(distances: List[float], bins: int = 5) -> List[DistanceBin]:
    if not distances:
        return []
    max_distance = max(distances)
    if max_distance == 0:
        return [DistanceBin(label="0 km", count=len(distances))]

    width = max_distance / bins if bins else max_distance
    counts = [0 for _ in range(bins)]
    for distance in distances:
        if width == 0:
            idx = 0
        else:
            idx = min(int(distance / width), bins - 1)
        counts[idx] += 1

    bins_payload: List[DistanceBin] = []
    for idx, count in enumerate(counts):
        start = idx * width
        end = max_distance if idx == bins - 1 else start + width
        label = f"{start:.1f}-{end:.1f} km"
        bins_payload.append(DistanceBin(label=label, count=count))
    return bins_payload
