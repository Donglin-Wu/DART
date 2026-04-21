"""
Initial sampling point generation.
"""

import numpy as np
from scipy.stats import qmc

from base import logger


def BuildBoundsFromData(xLow):
    """
    Build search bounds from low-fidelity data.
    """
    lowerBounds = xLow.min(axis=0)
    upperBounds = xLow.max(axis=0)
    bounds = [(float(low), float(high)) for low, high in zip(lowerBounds, upperBounds)]
    logger.info(f"Data bounds: {bounds}")
    return bounds


def LatinHypercubeSampling(bounds, sampleCount, randomState=None):
    """
    Latin hypercube sampling.
    """
    dimension = len(bounds)
    sampler = qmc.LatinHypercube(d=dimension, seed=randomState)
    unitSamples = sampler.random(n=sampleCount)
    lowerBounds = np.array([bound[0] for bound in bounds], dtype=float)
    upperBounds = np.array([bound[1] for bound in bounds], dtype=float)
    return qmc.scale(unitSamples, lowerBounds, upperBounds)


def CalculateEtaFromPoints(points):
    """
    Estimate the minimum spacing threshold from a point set.
    """
    if len(points) < 2:
        return 0.1

    dimension = points.shape[1]
    minDistance = float("inf")
    for firstIndex in range(len(points)):
        for secondIndex in range(firstIndex + 1, len(points)):
            distance = np.linalg.norm(points[firstIndex] - points[secondIndex])
            if distance < minDistance:
                minDistance = distance

    return max(0.001, minDistance / np.sqrt(dimension))


def SelectInitialPoints(minPoints, maxPoints):
    """
    Keep mutually distant points from extrema candidates.
    """
    allPoints = np.vstack([minPoints, maxPoints])
    if len(allPoints) == 0:
        return allPoints

    eta = CalculateEtaFromPoints(allPoints)
    selectedIndexes = [0]

    for pointIndex in range(1, len(allPoints)):
        minDistance = min(
            np.linalg.norm(allPoints[pointIndex] - allPoints[selectedIndex])
            for selectedIndex in selectedIndexes
        )
        if minDistance > eta:
            selectedIndexes.append(pointIndex)

    return allPoints[selectedIndexes]
