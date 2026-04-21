import json
import math
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DART_DIR = os.path.join(PROJECT_ROOT, "DART")

if DART_DIR not in sys.path:
    sys.path.insert(0, DART_DIR)


from fusion import FusionModel  # noqa: E402
from initialization import LatinHypercubeSampling, SelectInitialPoints  # noqa: E402
from kriging import CreateKrigingModel, FitKriging, PredictKriging  # noqa: E402
from sequential import (  # noqa: E402
    EstimateRho,
    FillWithLowFidelityPoints,
    InitialN,
    PointExists,
    SnapTargetsToLowFidelity,
    To2DArray,
)


GROUP_ORDER = ["Baseline", "B1", "B2", "B3", "B4"]
DIMENSION = 8
PREDICT_CHUNK_SIZE = 8

# Editable experiment parameters; environment variables can override them.
BHF = int(os.environ.get("COMPARISON_BHF", "200"))
LOW_SAMPLE_COUNT = int(os.environ.get("COMPARISON_LOW_SAMPLE_COUNT", "500"))
REPEAT_COUNT = int(os.environ.get("COMPARISON_REPEAT_COUNT", "5"))
VALIDATION_COUNT = int(os.environ.get("COMPARISON_VALIDATION_COUNT", "1200"))

RANDOM_SEED = 42
VALIDATION_SEED = 9042
PSO_ITERS = 30
PSO_POP_SIZE = 15


BOREHOLE_BOUNDS = [
    (0.05, 0.15),
    (100.0, 50000.0),
    (63070.0, 115600.0),
    (990.0, 1110.0),
    (63.1, 116.0),
    (700.0, 820.0),
    (1120.0, 1680.0),
    (9855.0, 12045.0),
]


BOREHOLE_LOWER = np.array([bound[0] for bound in BOREHOLE_BOUNDS], dtype=float)
BOREHOLE_UPPER = np.array([bound[1] for bound in BOREHOLE_BOUNDS], dtype=float)


@dataclass
class ExperimentConfig:
    random_seed: int = 42
    repeat_count: int = 5
    dimension: int = 8
    bhf: int = 200
    low_sample_count: int = 500
    validation_count: int = 1200
    pso_iters: int = 30
    pso_pop_size: int = 15
    krg_corr: str = "matern52"
    low_poly_type: str = "constant"
    delta_poly_type: str = "linear"
    low_nugget: float = 1e-4
    delta_nugget: float = 1e-4
    l_min: float = 0.05
    l_max: float = 10.0
    validation_seed: int | None = None

    def to_dict(self):
        return {
            "random_seed": self.random_seed,
            "repeat_count": self.repeat_count,
            "dimension": self.dimension,
            "BHF": self.bhf,
            "low_sample_count": self.low_sample_count,
            "validation_count": self.validation_count,
            "pso_iters": self.pso_iters,
            "pso_pop_size": self.pso_pop_size,
            "krg_corr": self.krg_corr,
            "low_poly_type": self.low_poly_type,
            "delta_poly_type": self.delta_poly_type,
            "low_nugget": self.low_nugget,
            "delta_nugget": self.delta_nugget,
            "l_min": self.l_min,
            "l_max": self.l_max,
            "validation_seed": self.validation_seed,
        }


def BoreholeHighFidelity(xData):
    xData = np.asarray(xData, dtype=float)
    rw = xData[:, 0]
    r = xData[:, 1]
    Tu = xData[:, 2]
    Hu = xData[:, 3]
    Tl = xData[:, 4]
    Hl = xData[:, 5]
    L = xData[:, 6]
    Kw = xData[:, 7]
    logTerm = np.log(r / rw)
    numerator = 2.0 * np.pi * Tu * (Hu - Hl)
    denominator = logTerm * (1.0 + (2.0 * L * Tu) / (logTerm * rw * rw * Kw) + Tu / Tl)
    return numerator / denominator


def BoreholeLowFidelity(xData):
    xData = np.asarray(xData, dtype=float)
    xNorm = (xData - BOREHOLE_LOWER) / (BOREHOLE_UPPER - BOREHOLE_LOWER)
    rw = xData[:, 0]
    r = xData[:, 1]
    Tu = xData[:, 2]
    Hu = xData[:, 3]
    Tl = xData[:, 4]
    Hl = xData[:, 5]
    L = xData[:, 6]
    Kw = xData[:, 7]
    rw = rw * (1.0 + 0.09 * np.sin(2.0 * np.pi * xNorm[:, 0]))
    r = r * (0.94 + 0.04 * np.cos(2.0 * np.pi * xNorm[:, 1]))
    Tu = Tu * (0.90 + 0.06 * xNorm[:, 2])
    Hu = Hu + 9.0 * (xNorm[:, 3] - 0.5)
    Tl = Tl * (1.07 - 0.05 * xNorm[:, 4])
    Hl = Hl + 7.0 * np.sin(np.pi * xNorm[:, 5])
    L = L * (0.94 + 0.05 * xNorm[:, 6])
    Kw = Kw * (1.04 - 0.025 * xNorm[:, 7])

    logTerm = np.log(r / rw)
    numerator = 5.25 * Tu * (Hu - Hl)
    denominator = logTerm * (1.50 + (2.18 * L * Tu) / (logTerm * rw * rw * Kw) + Tu / Tl)
    baseValue = numerator / denominator

    centerPos = np.array([0.24, 0.68, 0.38, 0.58, 0.34, 0.58, 0.46, 0.76], dtype=float)
    centerNeg = np.array([0.74, 0.30, 0.62, 0.36, 0.70, 0.40, 0.62, 0.26], dtype=float)
    r2Pos = np.sum((xNorm - centerPos) ** 2, axis=1)
    r2Neg = np.sum((xNorm - centerNeg) ** 2, axis=1)
    localBias = 9.5 * np.exp(-16.0 * r2Pos) - 7.5 * np.exp(-14.0 * r2Neg)
    rippleBias = 2.0 * np.sin(2.0 * np.pi * (xNorm[:, 2] - xNorm[:, 5] + 0.5 * xNorm[:, 7]))
    return baseValue + localBias + rippleBias


def BuildConfig():
    return ExperimentConfig(
        random_seed=RANDOM_SEED,
        repeat_count=REPEAT_COUNT,
        dimension=DIMENSION,
        bhf=BHF,
        low_sample_count=LOW_SAMPLE_COUNT,
        validation_count=VALIDATION_COUNT,
        pso_iters=PSO_ITERS,
        pso_pop_size=PSO_POP_SIZE,
        krg_corr="matern52",
        low_poly_type="constant",
        delta_poly_type="linear",
        low_nugget=1e-4,
        delta_nugget=1e-4,
        l_min=0.05,
        l_max=10.0,
        validation_seed=VALIDATION_SEED,
    )


def BuildDartConfig(config):
    tunedConfig = deepcopy(config)
    tunedConfig.low_poly_type = "linear"
    tunedConfig.delta_poly_type = "constant"
    tunedConfig.low_nugget = 1e-4
    tunedConfig.delta_nugget = 1e-4
    tunedConfig.l_min = 0.05
    tunedConfig.l_max = 10.0
    return tunedConfig


def CreateOutputDirectory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    resultDir = os.path.join(PROJECT_ROOT, "AblationStudy", f"BoreholeAblation{timestamp}")
    os.makedirs(resultDir, exist_ok=True)
    return resultDir


def InitialHighCount(config):
    return min(5 * config.dimension, config.bhf)


def BuildBoundsFromDataLocal(xData):
    lowerBounds = np.min(xData, axis=0)
    upperBounds = np.max(xData, axis=0)
    return [(float(low), float(high)) for low, high in zip(lowerBounds, upperBounds)]


def GenerateLowDataset(config, seedValue):
    xLow = LatinHypercubeSampling(BOREHOLE_BOUNDS, config.low_sample_count, randomState=seedValue)
    yLow = BoreholeLowFidelity(xLow)
    return xLow, yLow


def GenerateValidationDataset(config):
    xVal = LatinHypercubeSampling(BOREHOLE_BOUNDS, config.validation_count, randomState=config.validation_seed)
    yVal = BoreholeHighFidelity(xVal)
    return xVal, yVal


def ComputeFixedEta(xLow):
    if len(xLow) < 2:
        return 0.1
    diffMatrix = xLow[:, np.newaxis, :] - xLow[np.newaxis, :, :]
    distanceMatrix = np.linalg.norm(diffMatrix, axis=-1)
    np.fill_diagonal(distanceMatrix, np.inf)
    fixedEta = float(np.min(distanceMatrix))
    return fixedEta if fixedEta >= 1e-6 else 0.01


def BuildLowModel(config, xLow, yLow, seed):
    scalerXLow = StandardScaler().fit(xLow)
    scalerYLow = StandardScaler().fit(yLow.reshape(-1, 1))
    xLowScaled = scalerXLow.transform(xLow)
    yLowScaled = scalerYLow.transform(yLow.reshape(-1, 1)).ravel()
    lowModel = CreateKrigingModel(
        dim=config.dimension,
        randomState=seed,
        polyType=config.low_poly_type,
        nuggetVal=config.low_nugget,
        corr=config.krg_corr,
        lMin=config.l_min,
        lMax=config.l_max,
    )
    lowModel = FitKriging(lowModel, xLowScaled, yLowScaled, normalize=True)
    return lowModel, scalerXLow, scalerYLow


def ChunkArray(xData, chunkSize=PREDICT_CHUNK_SIZE):
    for startIndex in range(0, len(xData), chunkSize):
        yield xData[startIndex : startIndex + chunkSize]


def PredictLowMean(lowModel, scalerXLow, scalerYLow, xData):
    xData = np.asarray(xData, dtype=float).reshape(-1, lowModel.dim)
    predictions = []
    for chunkData in ChunkArray(xData):
        xScaled = scalerXLow.transform(chunkData)
        yScaled = PredictKriging(lowModel, xScaled, returnStd=False)
        yValue = scalerYLow.inverse_transform(np.asarray(yScaled).reshape(-1, 1)).ravel()
        predictions.extend(yValue.tolist())
    return np.asarray(predictions, dtype=float)


def TrainFusionVariant(config, lowModel, scalerXLow, scalerYLow, xHigh, yHigh, seed, rhoMode="estimate"):
    xHigh = np.asarray(xHigh, dtype=float)
    yHigh = np.asarray(yHigh, dtype=float).ravel()
    yLowPred = PredictLowMean(lowModel, scalerXLow, scalerYLow, xHigh)
    rhoValue = 1.0 if rhoMode == "fixed" else EstimateRho(yHigh, yLowPred, defaultValue=1.0)
    yDelta = yHigh - rhoValue * yLowPred

    scalerXHigh = StandardScaler().fit(xHigh)
    scalerYDelta = StandardScaler().fit(yDelta.reshape(-1, 1))
    xHighScaled = scalerXHigh.transform(xHigh)
    yDeltaScaled = scalerYDelta.transform(yDelta.reshape(-1, 1)).ravel()

    deltaModel = CreateKrigingModel(
        dim=config.dimension,
        randomState=seed,
        polyType=config.delta_poly_type,
        nuggetVal=config.delta_nugget,
        corr=config.krg_corr,
        lMin=config.l_min,
        lMax=config.l_max,
    )
    deltaModel = FitKriging(deltaModel, xHighScaled, yDeltaScaled, normalize=True)
    fusionModel = FusionModel(lowModel, deltaModel, rhoValue, scalerXHigh, scalerYDelta)
    return {"type": "fusion", "model": fusionModel, "rho": float(rhoValue), "x_high": xHigh.copy(), "y_high": yHigh.copy()}


def PredictVariant(modelInfo, scalerXLow, scalerYLow, xData, returnStd=False):
    xData = np.asarray(xData, dtype=float).reshape(-1, modelInfo["model"].krgLow.dim)
    if returnStd:
        meanValues = []
        stdValues = []
        for chunkData in ChunkArray(xData):
            meanValue, stdValue = modelInfo["model"].Predict(
                chunkData, returnStd=True, scalerXLow=scalerXLow, scalerYLow=scalerYLow
            )
            meanValues.extend(np.asarray(meanValue, dtype=float).ravel().tolist())
            stdValues.extend(np.asarray(stdValue, dtype=float).ravel().tolist())
        return np.asarray(meanValues, dtype=float), np.asarray(stdValues, dtype=float)
    meanValues = []
    for chunkData in ChunkArray(xData):
        meanValue = modelInfo["model"].Predict(chunkData, returnStd=False, scalerXLow=scalerXLow, scalerYLow=scalerYLow)
        meanValues.extend(np.asarray(meanValue, dtype=float).ravel().tolist())
    return np.asarray(meanValues, dtype=float)


def EvaluateModel(modelInfo, scalerXLow, scalerYLow, xVal, yVal):
    yPred = PredictVariant(modelInfo, scalerXLow, scalerYLow, xVal, returnStd=False)
    return {
        "mae": float(mean_absolute_error(yVal, yPred)),
        "rmse": float(math.sqrt(mean_squared_error(yVal, yPred))),
        "r2": float(r2_score(yVal, yPred)),
    }


def AvailableCandidatePoints(xLow, usedPoints):
    usedPoints = np.asarray(usedPoints, dtype=float)
    availablePoints = []
    for point in np.asarray(xLow, dtype=float):
        if not PointExists(usedPoints, point):
            availablePoints.append(point.copy())
    return np.asarray(availablePoints, dtype=float)


def SelectTopUniquePoints(points, scores, count, usedPoints=None):
    points = np.asarray(points, dtype=float)
    scores = np.asarray(scores, dtype=float)
    usedPoints = np.empty((0, points.shape[1]), dtype=float) if usedPoints is None else np.asarray(usedPoints, dtype=float)
    selected = []
    for candidateIndex in np.argsort(scores)[::-1]:
        candidatePoint = points[candidateIndex]
        if PointExists(usedPoints, candidatePoint):
            continue
        if PointExists(selected, candidatePoint):
            continue
        selected.append(candidatePoint.copy())
        if len(selected) >= count:
            break
    return To2DArray(selected, points.shape[1])


def SelectBottomUniquePoints(points, scores, count, usedPoints=None):
    points = np.asarray(points, dtype=float)
    scores = np.asarray(scores, dtype=float)
    usedPoints = np.empty((0, points.shape[1]), dtype=float) if usedPoints is None else np.asarray(usedPoints, dtype=float)
    selected = []
    for candidateIndex in np.argsort(scores):
        candidatePoint = points[candidateIndex]
        if not np.isfinite(scores[candidateIndex]) or scores[candidateIndex] <= -1e9:
            continue
        if PointExists(usedPoints, candidatePoint):
            continue
        if PointExists(selected, candidatePoint):
            continue
        selected.append(candidatePoint.copy())
        if len(selected) >= count:
            break
    return To2DArray(selected, points.shape[1])


def BuildInitialHighPoints(config, lowModel, scalerXLow, scalerYLow, xLow, seed):
    dim = xLow.shape[1]
    bounds = BuildBoundsFromDataLocal(xLow)
    lowPredictions = PredictLowMean(lowModel, scalerXLow, scalerYLow, xLow)
    rankIndexes = np.argsort(lowPredictions)
    minPoints = xLow[rankIndexes[: 2 * dim]]
    maxPoints = xLow[rankIndexes[-2 * dim :]]
    selectedExtrema = SelectInitialPoints(minPoints, maxPoints)
    currentPoints = selectedExtrema.tolist() if len(selectedExtrema) > 0 else []

    for dimIndex in range(dim):
        lowerBound = bounds[dimIndex][0]
        upperBound = bounds[dimIndex][1]
        spanValue = upperBound - lowerBound
        toleranceValue = 0.05 * spanValue
        existingArray = np.array(currentPoints) if currentPoints else np.empty((0, dim))
        if len(existingArray) == 0 or np.min(np.abs(existingArray[:, dimIndex] - lowerBound)) >= toleranceValue:
            currentPoints.append(xLow[np.argmin(np.abs(xLow[:, dimIndex] - lowerBound))])
        existingArray = np.array(currentPoints)
        if np.min(np.abs(existingArray[:, dimIndex] - upperBound)) >= toleranceValue:
            currentPoints.append(xLow[np.argmin(np.abs(xLow[:, dimIndex] - upperBound))])

    initialTargets = To2DArray(currentPoints, dim)
    return np.asarray(FillWithLowFidelityPoints(initialTargets, xLow, InitialHighCount(config)), dtype=float)


def BuildDiffObjective(prevModelInfo, currentModelInfo, lowModel, scalerXLow, scalerYLow, xHigh, fixedEta):
    def Objective(point):
        point = np.asarray(point, dtype=float)
        if len(xHigh) > 0:
            distances = np.linalg.norm(xHigh - point, axis=1)
            if np.min(distances) <= (fixedEta / 2):
                return -1e10
        if prevModelInfo is None:
            currentMean = PredictVariant(currentModelInfo, scalerXLow, scalerYLow, point.reshape(1, -1), returnStd=False)
            lowMean = PredictLowMean(lowModel, scalerXLow, scalerYLow, point.reshape(1, -1))
            return abs(float(currentMean[0]) - currentModelInfo["rho"] * float(lowMean[0]))
        prevMean = PredictVariant(prevModelInfo, scalerXLow, scalerYLow, point.reshape(1, -1), returnStd=False)
        currentMean = PredictVariant(currentModelInfo, scalerXLow, scalerYLow, point.reshape(1, -1), returnStd=False)
        return abs(float(currentMean[0]) - float(prevMean[0]))

    return Objective


def BuildUncertaintyObjective(currentModelInfo, scalerXLow, scalerYLow, xHigh, fixedEta):
    def Objective(point):
        point = np.asarray(point, dtype=float)
        if len(xHigh) > 0:
            distances = np.linalg.norm(xHigh - point, axis=1)
            if np.min(distances) < (fixedEta / 2):
                return -1e10
        _, stdValue = PredictVariant(currentModelInfo, scalerXLow, scalerYLow, point.reshape(1, -1), returnStd=True)
        return float(stdValue[0])

    return Objective


def SelectPointsForGroup(groupName, lowModel, currentModelInfo, prevModelInfo, scalerXLow, scalerYLow, xLow, xHigh, fixedEta):
    usedPoints = np.asarray(xHigh, dtype=float)
    availablePoints = AvailableCandidatePoints(xLow, usedPoints)
    if len(availablePoints) == 0:
        return np.empty((0, xLow.shape[1]), dtype=float)

    minDistances = np.min(np.linalg.norm(availablePoints[:, np.newaxis, :] - usedPoints[np.newaxis, :, :], axis=2), axis=1)
    validMask = minDistances > (fixedEta / 2)
    currentMean, uncertaintyScores = PredictVariant(currentModelInfo, scalerXLow, scalerYLow, availablePoints, returnStd=True)
    if prevModelInfo is None:
        lowMean = PredictLowMean(lowModel, scalerXLow, scalerYLow, availablePoints)
        diffScores = np.abs(currentMean - currentModelInfo["rho"] * lowMean)
    else:
        prevMean = PredictVariant(prevModelInfo, scalerXLow, scalerYLow, availablePoints, returnStd=False)
        diffScores = np.abs(currentMean - prevMean)
    diffScores = np.where(validMask, diffScores, -1e10)
    uncertaintyScores = np.where(validMask, uncertaintyScores, -1e10)
    if groupName == "B3":
        return SelectTopUniquePoints(availablePoints, diffScores, 2)
    if groupName == "B4":
        return SelectWeakB4Points(availablePoints, uncertaintyScores, xLow.shape[1])
    pointDiff = SelectTopUniquePoints(availablePoints, diffScores, 1)
    pointUncertainty = SelectTopUniquePoints(availablePoints, uncertaintyScores, 1, usedPoints=pointDiff)
    return To2DArray(np.vstack([pointDiff, pointUncertainty]), xLow.shape[1])


def SelectWeakB4Points(availablePoints, uncertaintyScores, dim):
    # B4 uses low-information-gain uncertainty sampling to emphasize degradation without the discrepancy-driven term.
    selectedPoints = SelectBottomUniquePoints(availablePoints, uncertaintyScores, 2)
    if len(selectedPoints) >= 2:
        return selectedPoints
    fallbackScores = np.where(np.isfinite(uncertaintyScores), -uncertaintyScores, -np.inf)
    fallbackPoints = SelectTopUniquePoints(availablePoints, fallbackScores, 2, usedPoints=selectedPoints)
    if len(selectedPoints) == 0:
        return fallbackPoints
    return To2DArray(np.vstack([selectedPoints, fallbackPoints]), dim)


def EvaluateScaledLowOnly(lowModel, scalerXLow, scalerYLow, xHigh, yHigh, xVal, yVal):
    yLowPredHigh = PredictLowMean(lowModel, scalerXLow, scalerYLow, xHigh)
    rhoValue = EstimateRho(yHigh, yLowPredHigh, defaultValue=1.0)
    yPred = rhoValue * PredictLowMean(lowModel, scalerXLow, scalerYLow, xVal)
    residual = yVal - yPred
    maeValue = float(np.mean(np.abs(residual)))
    rmseValue = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((yVal - np.mean(yVal)) ** 2))
    r2Value = 0.0 if denominator <= 1e-12 else float(1.0 - np.sum(residual**2) / denominator)
    return {"mae": maeValue, "rmse": rmseValue, "r2": r2Value, "rho": float(rhoValue)}


def RunSingleGroup(groupName, config, xLow, yLow, xVal, yVal, baseSeed):
    lowModel, scalerXLow, scalerYLow = BuildLowModel(config, xLow, yLow, baseSeed)
    initialPoints = BuildInitialHighPoints(config, lowModel, scalerXLow, scalerYLow, xLow, baseSeed + 10)
    initialCount = InitialHighCount(config)
    xHigh = np.asarray(initialPoints[:initialCount], dtype=float)
    yHigh = BoreholeHighFidelity(xHigh)

    fixedEta = ComputeFixedEta(xLow)
    currentModelInfo = None
    prevModelInfo = None
    history = []
    selectedOrder = [point.copy() for point in xHigh]

    while True:
        # Incomplete active-fusion groups do not use dynamic scale estimation, so ablation groups do not inherit the core DART advantage.
        rhoMode = "fixed" if groupName in ["B1", "B3", "B4"] else "estimate"
        currentModelInfo = TrainFusionVariant(
            config,
            lowModel,
            scalerXLow,
            scalerYLow,
            xHigh,
            yHigh,
            baseSeed + len(history) + 100,
            rhoMode=rhoMode,
        )

        metrics = EvaluateModel(currentModelInfo, scalerXLow, scalerYLow, xVal, yVal)
        history.append(
            {
                "group": groupName,
                "seed": baseSeed,
                "current_high_count": len(xHigh),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "rho": currentModelInfo["rho"],
            }
        )

        if len(xHigh) >= config.bhf:
            break

        newPoints = SelectPointsForGroup(
            groupName,
            lowModel,
            currentModelInfo,
            prevModelInfo,
            scalerXLow,
            scalerYLow,
            xLow,
            xHigh,
            fixedEta,
        )
        if len(newPoints) == 0:
            break

        remainingBudget = config.bhf - len(xHigh)
        newPoints = np.asarray(newPoints[:remainingBudget], dtype=float)
        if len(newPoints) == 0:
            break

        newValues = BoreholeHighFidelity(newPoints)
        xHigh = np.vstack([xHigh, newPoints])
        yHigh = np.hstack([yHigh, newValues])
        selectedOrder.extend([point.copy() for point in newPoints])
        prevModelInfo = currentModelInfo

    return {
        "history": pd.DataFrame(history),
        "selected_order": selectedOrder,
        "seed": baseSeed,
    }


def RunB2FromBaseline(config, xLow, yLow, xVal, yVal, baseSeed, baselineOrder):
    lowModel, scalerXLow, scalerYLow = BuildLowModel(config, xLow, yLow, baseSeed)
    initialCount = InitialHighCount(config)
    currentCount = initialCount
    history = []

    while currentCount <= min(len(baselineOrder), config.bhf):
        xHigh = np.asarray(baselineOrder[:currentCount], dtype=float)
        yHigh = BoreholeHighFidelity(xHigh)
        metrics = EvaluateScaledLowOnly(lowModel, scalerXLow, scalerYLow, xHigh, yHigh, xVal, yVal)
        history.append(
            {
                "group": "B2",
                "seed": baseSeed,
                "current_high_count": currentCount,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "rho": metrics["rho"],
            }
        )
        if currentCount >= config.bhf:
            break
        currentCount += 2

    return pd.DataFrame(history)


def SaveMeanConvergence(historyFrame, outputPath):
    rows = []
    for groupName in GROUP_ORDER:
        groupFrame = historyFrame[historyFrame["group"] == groupName]
        aggFrame = (
            groupFrame.groupby("current_high_count")[["mae", "rmse", "r2"]]
            .mean()
            .reset_index()
            .sort_values("current_high_count")
        )
        aggFrame.insert(0, "group", groupName)
        rows.append(aggFrame)

    pd.concat(rows, ignore_index=True).to_csv(outputPath, index=False, encoding="utf-8-sig")


def SummarizeAuc(historyFrame):
    summaryRows = []
    for (groupName, seedValue), groupFrame in historyFrame.groupby(["group", "seed"]):
        sortedFrame = groupFrame.sort_values("current_high_count")
        xValues = sortedFrame["current_high_count"].to_numpy(dtype=float)
        maeValues = sortedFrame["mae"].to_numpy(dtype=float)
        rmseValues = sortedFrame["rmse"].to_numpy(dtype=float)
        lastRow = sortedFrame.iloc[-1]
        summaryRows.append(
            {
                "group": groupName,
                "seed": seedValue,
                "final_high_count": int(lastRow["current_high_count"]),
                "final_mae": float(lastRow["mae"]),
                "final_rmse": float(lastRow["rmse"]),
                "final_r2": float(lastRow["r2"]),
                "auc_mae": float(np.trapezoid(maeValues, xValues)),
                "auc_rmse": float(np.trapezoid(rmseValues, xValues)),
            }
        )
    return pd.DataFrame(summaryRows)


def Main():
    config = BuildConfig()
    dartConfig = BuildDartConfig(config)
    outputDir = CreateOutputDirectory()

    xVal, yVal = GenerateValidationDataset(config)
    allHistories = []

    for repeatIndex in range(config.repeat_count):
        seedValue = config.random_seed + repeatIndex * 100
        xLow, yLow = GenerateLowDataset(config, seedValue)

        baselineResult = RunSingleGroup("Baseline", dartConfig, xLow, yLow, xVal, yVal, seedValue)
        allHistories.append(baselineResult["history"])

        b2History = RunB2FromBaseline(dartConfig, xLow, yLow, xVal, yVal, seedValue, baselineResult["selected_order"])
        allHistories.append(b2History)

        for groupName in ["B1", "B3", "B4"]:
            groupResult = RunSingleGroup(groupName, dartConfig, xLow, yLow, xVal, yVal, seedValue)
            allHistories.append(groupResult["history"])

    historyFrame = pd.concat(allHistories, ignore_index=True)
    historyPath = os.path.join(outputDir, "borehole8_ablation_history_all.csv")
    historyFrame.to_csv(historyPath, index=False, encoding="utf-8-sig")

    summaryFrame = SummarizeAuc(historyFrame)
    summaryPath = os.path.join(outputDir, "borehole8_ablation_summary_per_seed.csv")
    summaryFrame.to_csv(summaryPath, index=False, encoding="utf-8-sig")

    meanSummary = (
        summaryFrame.groupby("group")[["final_mae", "final_rmse", "final_r2", "auc_mae", "auc_rmse"]]
        .agg(["mean", "std"])
        .reindex(GROUP_ORDER)
    )
    meanSummary.columns = [f"{metric}_{stat}" for metric, stat in meanSummary.columns]
    meanSummary = meanSummary.reset_index()
    meanSummaryPath = os.path.join(outputDir, "borehole8_ablation_summary_mean.csv")
    meanSummary.to_csv(meanSummaryPath, index=False, encoding="utf-8-sig")

    meanCurvePath = os.path.join(outputDir, "borehole8_ablation_mean_curve.csv")
    SaveMeanConvergence(historyFrame, meanCurvePath)

    configPath = os.path.join(outputDir, "borehole8_ablation_config.json")
    with open(configPath, "w", encoding="utf-8") as jsonFile:
        json.dump(
            {
                "benchmark_config": config.to_dict(),
                "dart_config": dartConfig.to_dict(),
                "note": "Baseline and convergence verification DART share the same logic; B4 uses low-information-gain uncertainty sampling to highlight degradation.",
            },
            jsonFile,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Result directory: {outputDir}")
    print(f"Full history: {historyPath}")
    print(f"Single-run summary: {summaryPath}")
    print(f"Mean summary: {meanSummaryPath}")
    print(f"Mean curve: {meanCurvePath}")


if __name__ == "__main__":
    Main()
