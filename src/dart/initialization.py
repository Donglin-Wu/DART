"""
初始采样点生成。
"""

import numpy as np
from scipy.stats import qmc

from .base import logger


def BuildBoundsFromData(xLow):
    """
    根据低精度数据构造搜索边界。
    """
    lowerBounds = xLow.min(axis=0)
    upperBounds = xLow.max(axis=0)
    bounds = [(float(low), float(high)) for low, high in zip(lowerBounds, upperBounds)]
    logger.info(f"数据边界: {bounds}")
    return bounds


def LatinHypercubeSampling(bounds, sampleCount, randomState=None):
    """
    拉丁超立方采样。
    """
    dimension = len(bounds)
    sampler = qmc.LatinHypercube(d=dimension, seed=randomState)
    unitSamples = sampler.random(n=sampleCount)
    lowerBounds = np.array([bound[0] for bound in bounds], dtype=float)
    upperBounds = np.array([bound[1] for bound in bounds], dtype=float)
    return qmc.scale(unitSamples, lowerBounds, upperBounds)


def CalculateEtaFromPoints(points):
    """
    根据点集估计最小间距阈值。
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
    从极值候选点中保留彼此距离较远的点。
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


# Compatibility aliases for legacy scripts.
build_bounds_from_data = BuildBoundsFromData
latin_hypercube_sampling = LatinHypercubeSampling
select_initial_points = SelectInitialPoints
