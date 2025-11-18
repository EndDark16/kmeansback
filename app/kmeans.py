from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import List, Sequence, Tuple


Point = Tuple[float, float]


@dataclass
class KMeansResult:
    centroids: List[Point]
    assignments: List[int]
    iterations: int


def euclidean_distance_sq(p1: Point, p2: Point) -> float:
    """Return the squared Euclidean distance between two 2D points."""
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def initialize_centroids(points: Sequence[Point], k: int, rng: Random) -> List[Point]:
    """Randomly choose k distinct points to seed the centroids."""
    if k > len(points):
        raise ValueError("k cannot be greater than the number of points")
    indices = rng.sample(range(len(points)), k)
    return [points[i] for i in indices]


def assign_points(points: Sequence[Point], centroids: Sequence[Point]) -> List[int]:
    """Assign each point to the closest centroid."""
    assignments: List[int] = []
    for point in points:
        min_idx = 0
        min_dist = float("inf")
        for idx, centroid in enumerate(centroids):
            dist = euclidean_distance_sq(point, centroid)
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
        assignments.append(min_idx)
    return assignments


def recompute_centroids(
    points: Sequence[Point], assignments: Sequence[int], k: int
) -> List[Point]:
    """Recompute centroids as the mean position of their assigned points."""
    sums = [[0.0, 0.0] for _ in range(k)]
    counts = [0 for _ in range(k)]
    for point, cluster in zip(points, assignments):
        sums[cluster][0] += point[0]
        sums[cluster][1] += point[1]
        counts[cluster] += 1

    new_centroids: List[Point] = []
    for idx in range(k):
        if counts[idx] == 0:
            # Caller is responsible for replacing empty centroids.
            new_centroids.append((math.nan, math.nan))
            continue
        new_centroids.append(
            (sums[idx][0] / counts[idx], sums[idx][1] / counts[idx]),
        )
    return new_centroids


def replace_empty_centroids(
    centroids: Sequence[Point],
    points: Sequence[Point],
    assignments: Sequence[int],
    rng: Random,
) -> List[Point]:
    """If a centroid received no points, re-seed it with a random point."""
    counts = [0 for _ in centroids]
    for cluster in assignments:
        counts[cluster] += 1

    new_centroids: List[Point] = list(centroids)
    for idx, centroid in enumerate(centroids):
        if not math.isfinite(centroid[0]) or counts[idx] == 0:
            new_centroids[idx] = points[rng.randrange(len(points))]
    return new_centroids


def run_kmeans(
    points: Sequence[Point],
    k: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    rng: Random | None = None,
) -> KMeansResult:
    """
    Run vanilla K-Means and return the final centroids plus assignments.

    Args:
        points: 2D points for clustering.
        k: number of desired centroids.
        max_iter: iteration cap.
        tol: convergence tolerance in coordinate distance.
        rng: optional RNG for deterministic runs.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if len(points) == 0:
        raise ValueError("At least one point is required")

    rng = rng or Random()
    centroids = initialize_centroids(points, k, rng)

    assignments = [0 for _ in points]
    for iteration in range(1, max_iter + 1):
        assignments = assign_points(points, centroids)
        raw_new_centroids = recompute_centroids(points, assignments, k)
        new_centroids = replace_empty_centroids(
            raw_new_centroids,
            points,
            assignments,
            rng,
        )

        deltas = [
            math.sqrt(euclidean_distance_sq(old, new))
            for old, new in zip(centroids, new_centroids)
        ]
        centroids = new_centroids
        if max(deltas) <= tol:
            return KMeansResult(centroids=centroids, assignments=assignments, iterations=iteration)

    return KMeansResult(centroids=centroids, assignments=assignments, iterations=max_iter)
