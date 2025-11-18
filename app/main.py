from __future__ import annotations

import json
import os
from pathlib import Path
from random import Random
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .kmeans import KMeansResult, run_kmeans
from .schemas import (
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
        return ["*"]
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
    return KMeansResponse(
        neighborhoods=neighborhoods,
        hospitals=hospitals,
        assignments=result.assignments,
        iterations=result.iterations,
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
