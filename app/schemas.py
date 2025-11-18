from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class KMeansRequest(BaseModel):
    m: int = Field(..., gt=0, description="City grid size (km).")
    n: int = Field(..., gt=0, description="Number of simulated neighborhoods.")
    k: int = Field(..., gt=0, description="Number of hospitals to compute.")
    max_iter: int = Field(100, gt=0, le=1000, description="Maximum K-Means iterations.")
    tol: float = Field(1e-4, ge=0, description="Convergence tolerance for centroids.")
    seed: int | None = Field(
        default=None,
        description="Optional random seed to reproduce simulations.",
    )


class Neighborhood(BaseModel):
    id: int
    x: float
    y: float


class Hospital(BaseModel):
    id: int
    x: float
    y: float


class KMeansResponse(BaseModel):
    neighborhoods: List[Neighborhood]
    hospitals: List[Hospital]
    assignments: List[int]
    iterations: int


class PretrainedModelResponse(BaseModel):
    k: int
    hospitals: List[List[float]]
    description: str
