import json
import os
import sys
from concurrent.futures.process import BrokenProcessPool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERGENCE_DIR = os.path.join(PROJECT_ROOT, "ConvergenceVerification")
AblationStudy_DIR = os.path.join(PROJECT_ROOT, "AblationStudy")
DART_DIR = os.path.join(PROJECT_ROOT, "DART")

for searchPath in [CONVERGENCE_DIR, AblationStudy_DIR, DART_DIR]:
    if searchPath not in sys.path:
        sys.path.insert(0, searchPath)


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


METHOD_ORDER = ["DART", "SF-Kriging", "Co-Kriging", "Hierarchical Kriging"]
DIMENSION = 8
DEFAULT_PREDICT_CHUNK_SIZE = 8

# Editable experiment parameters
BHF = int(os.environ.get("COMPARISON_BHF", "200"))
LOW_SAMPLE_COUNT = int(os.environ.get("COMPARISON_LOW_SAMPLE_COUNT", "500"))
REPEAT_COUNT = int(os.environ.get("COMPARISON_REPEAT_COUNT", "5"))
VALIDATION_COUNT = int(os.environ.get("COMPARISON_VALIDATION_COUNT", "1200"))

# Other default parameters
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


def BuildBenchmarkConfig():
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


def GenerateInitialHighPoints(config, xLow, seedValue):
    dim = xLow.shape[1]
    bounds = BuildBoundsFromDataLocal(xLow)
    initialCount = InitialHighCount(config)
    initialTargets = LatinHypercubeSampling(bounds, initialCount, randomState=seedValue)
    snappedPoints = SnapTargetsToLowFidelity(initialTargets, xLow, usedPoints=None)
    snappedPoints = [point for point in snappedPoints if point is not None]
    initialPoints = FillWithLowFidelityPoints(To2DArray(snappedPoints, dim), xLow, initialCount)
    return np.asarray(initialPoints, dtype=float)


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


def PredictLowMean(lowModel, scalerXLow, scalerYLow, xData):
    xData = np.asarray(xData, dtype=float).reshape(-1, lowModel.dim)
    predictions = []
    for row in xData:
        xScaled = scalerXLow.transform(row.reshape(1, -1))
        yScaled = PredictKriging(lowModel, xScaled, returnStd=False)
        yValue = scalerYLow.inverse_transform(np.asarray(yScaled).reshape(-1, 1)).ravel()[0]
        predictions.append(float(yValue))
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
    xData = np.asarray(xData, dtype=float)
    meanValues = []
    stdValues = []
    for row in xData:
        if returnStd:
            meanValue, stdValue = modelInfo["model"].Predict(
                row.reshape(1, -1), returnStd=True, scalerXLow=scalerXLow, scalerYLow=scalerYLow
            )
            meanValues.append(float(np.asarray(meanValue).ravel()[0]))
            stdValues.append(float(np.asarray(stdValue).ravel()[0]))
        else:
            meanValue = modelInfo["model"].Predict(
                row.reshape(1, -1), returnStd=False, scalerXLow=scalerXLow, scalerYLow=scalerYLow
            )
            meanValues.append(float(np.asarray(meanValue).ravel()[0]))
    if returnStd:
        return np.asarray(meanValues, dtype=float), np.asarray(stdValues, dtype=float)
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

    diffObjective = BuildDiffObjective(prevModelInfo, currentModelInfo, lowModel, scalerXLow, scalerYLow, usedPoints, fixedEta)
    uncertaintyObjective = BuildUncertaintyObjective(currentModelInfo, scalerXLow, scalerYLow, usedPoints, fixedEta)
    diffScores = np.array([diffObjective(point) for point in availablePoints], dtype=float)
    uncertaintyScores = np.array([uncertaintyObjective(point) for point in availablePoints], dtype=float)
    if groupName == "B3":
        return SelectTopUniquePoints(availablePoints, diffScores, 2)
    if groupName == "B4":
        return SelectTopUniquePoints(availablePoints, uncertaintyScores, 2)
    pointDiff = SelectTopUniquePoints(availablePoints, diffScores, 1)
    pointUncertainty = SelectTopUniquePoints(availablePoints, uncertaintyScores, 1, usedPoints=pointDiff)
    return To2DArray(np.vstack([pointDiff, pointUncertainty]), xLow.shape[1])


def ChunkArray(xData, chunkSize):
    for startIndex in range(0, len(xData), chunkSize):
        yield xData[startIndex : startIndex + chunkSize]


def BuildResidualModel(config, xHigh, residualValues, seed):
    scalerX = StandardScaler().fit(xHigh)
    scalerY = StandardScaler().fit(residualValues.reshape(-1, 1))
    xScaled = scalerX.transform(xHigh)
    yScaled = scalerY.transform(residualValues.reshape(-1, 1)).ravel()

    residualModel = CreateKrigingModel(
        dim=config.dimension,
        randomState=seed,
        polyType=config.delta_poly_type,
        nuggetVal=config.delta_nugget,
        corr=config.krg_corr,
        lMin=config.l_min,
        lMax=config.l_max,
    )
    residualModel = FitKriging(residualModel, xScaled, yScaled, normalize=True)
    return {"model": residualModel, "scalerX": scalerX, "scalerY": scalerY}


def PredictResidualModel(modelInfo, xData, returnStd=False):
    xData = np.asarray(xData, dtype=float).reshape(-1, modelInfo["model"].dim)
    meanValues = []
    stdValues = []
    scaleValue = float(modelInfo["scalerY"].scale_[0])

    for chunkData in ChunkArray(xData, DEFAULT_PREDICT_CHUNK_SIZE):
        xScaled = modelInfo["scalerX"].transform(chunkData)
        if returnStd:
            yPredScaled, yStdScaled = PredictKriging(modelInfo["model"], xScaled, returnStd=True)
            meanChunk = modelInfo["scalerY"].inverse_transform(np.asarray(yPredScaled).reshape(-1, 1)).ravel()
            stdChunk = np.asarray(yStdScaled, dtype=float).ravel() * scaleValue
            meanValues.extend(meanChunk.tolist())
            stdValues.extend(stdChunk.tolist())
        else:
            yPredScaled = PredictKriging(modelInfo["model"], xScaled, returnStd=False)
            meanChunk = modelInfo["scalerY"].inverse_transform(np.asarray(yPredScaled).reshape(-1, 1)).ravel()
            meanValues.extend(meanChunk.tolist())

    if returnStd:
        return np.asarray(meanValues, dtype=float), np.asarray(stdValues, dtype=float)
    return np.asarray(meanValues, dtype=float)


def BuildSingleFidelityModel(config, xHigh, yHigh, seed):
    scalerX = StandardScaler().fit(xHigh)
    scalerY = StandardScaler().fit(yHigh.reshape(-1, 1))
    xScaled = scalerX.transform(xHigh)
    yScaled = scalerY.transform(yHigh.reshape(-1, 1)).ravel()

    sfModel = CreateKrigingModel(
        dim=config.dimension,
        randomState=seed,
        polyType=config.low_poly_type,
        nuggetVal=config.low_nugget,
        corr=config.krg_corr,
        lMin=config.l_min,
        lMax=config.l_max,
    )
    sfModel = FitKriging(sfModel, xScaled, yScaled, normalize=True)
    return {"model": sfModel, "scalerX": scalerX, "scalerY": scalerY}


def PredictSingleFidelity(modelInfo, xData, returnStd=False):
    xData = np.asarray(xData, dtype=float).reshape(-1, modelInfo["model"].dim)
    meanValues = []
    stdValues = []
    scaleValue = float(modelInfo["scalerY"].scale_[0])

    for chunkData in ChunkArray(xData, DEFAULT_PREDICT_CHUNK_SIZE):
        xScaled = modelInfo["scalerX"].transform(chunkData)
        if returnStd:
            yPredScaled, yStdScaled = PredictKriging(modelInfo["model"], xScaled, returnStd=True)
            meanChunk = modelInfo["scalerY"].inverse_transform(np.asarray(yPredScaled).reshape(-1, 1)).ravel()
            stdChunk = np.asarray(yStdScaled, dtype=float).ravel() * scaleValue
            meanValues.extend(meanChunk.tolist())
            stdValues.extend(stdChunk.tolist())
        else:
            yPredScaled = PredictKriging(modelInfo["model"], xScaled, returnStd=False)
            meanChunk = modelInfo["scalerY"].inverse_transform(np.asarray(yPredScaled).reshape(-1, 1)).ravel()
            meanValues.extend(meanChunk.tolist())

    if returnStd:
        return np.asarray(meanValues, dtype=float), np.asarray(stdValues, dtype=float)
    return np.asarray(meanValues, dtype=float)


def BuildCoKrigingModel(config, lowModel, scalerXLow, scalerYLow, xHigh, yHigh, seed):
    lowPredHigh = PredictLowMean(lowModel, scalerXLow, scalerYLow, xHigh)
    rhoValue = EstimateRho(yHigh, lowPredHigh, defaultValue=1.0)
    residualValues = yHigh - rhoValue * lowPredHigh
    residualModel = BuildResidualModel(config, xHigh, residualValues, seed)
    return {
        "lowModel": lowModel,
        "scalerXLow": scalerXLow,
        "scalerYLow": scalerYLow,
        "rho": float(rhoValue),
        "residualModel": residualModel,
    }


def PredictCoKriging(modelInfo, xData, returnStd=False):
    lowMean = PredictLowMean(modelInfo["lowModel"], modelInfo["scalerXLow"], modelInfo["scalerYLow"], xData)
    if returnStd:
        residualMean, residualStd = PredictResidualModel(modelInfo["residualModel"], xData, returnStd=True)
        return modelInfo["rho"] * lowMean + residualMean, residualStd
    return modelInfo["rho"] * lowMean + PredictResidualModel(modelInfo["residualModel"], xData, returnStd=False)


def BuildHierarchicalKrigingModel(config, lowModel, scalerXLow, scalerYLow, xHigh, yHigh, seed):
    lowPredHigh = PredictLowMean(lowModel, scalerXLow, scalerYLow, xHigh)
    lowMeanValue = float(np.mean(lowPredHigh))
    highMeanValue = float(np.mean(yHigh))
    lowCentered = lowPredHigh - lowMeanValue
    highCentered = yHigh - highMeanValue
    denominator = float(np.dot(lowCentered, lowCentered))
    beta1 = 0.0 if denominator <= 1e-12 else float(np.dot(lowCentered, highCentered) / denominator)
    beta0 = highMeanValue - beta1 * lowMeanValue
    residualValues = yHigh - (beta0 + beta1 * lowPredHigh)
    residualModel = BuildResidualModel(config, xHigh, residualValues, seed)
    return {
        "lowModel": lowModel,
        "scalerXLow": scalerXLow,
        "scalerYLow": scalerYLow,
        "beta": np.asarray([beta0, beta1], dtype=float),
        "residualModel": residualModel,
    }


def PredictHierarchicalKriging(modelInfo, xData, returnStd=False):
    lowMean = PredictLowMean(modelInfo["lowModel"], modelInfo["scalerXLow"], modelInfo["scalerYLow"], xData)
    trendMean = modelInfo["beta"][0] + modelInfo["beta"][1] * lowMean
    if returnStd:
        residualMean, residualStd = PredictResidualModel(modelInfo["residualModel"], xData, returnStd=True)
        return trendMean + residualMean, residualStd
    return trendMean + PredictResidualModel(modelInfo["residualModel"], xData, returnStd=False)


def EvaluateWithPredictor(predictorFn, modelInfo, xVal, yVal):
    yPred = predictorFn(modelInfo, xVal, returnStd=False)
    return float(mean_absolute_error(yVal, yPred))


def SelectPointByStd(predictorFn, modelInfo, xLow, xHigh, fixedEta):
    availablePoints = []
    xHigh = np.asarray(xHigh, dtype=float)

    for candidatePoint in np.asarray(xLow, dtype=float):
        if PointExists(xHigh, candidatePoint):
            continue
        availablePoints.append(candidatePoint.copy())

    if not availablePoints:
        return None

    availablePoints = np.asarray(availablePoints, dtype=float)
    minDistances = np.min(np.linalg.norm(availablePoints[:, np.newaxis, :] - xHigh[np.newaxis, :, :], axis=2), axis=1)
    validPoints = availablePoints[minDistances > (fixedEta / 2.0)]
    candidatePoints = validPoints if len(validPoints) > 0 else availablePoints
    _, stdValues = predictorFn(modelInfo, candidatePoints, returnStd=True)
    return candidatePoints[int(np.argmax(stdValues))].copy()


def RunDartBaseline(config, xLow, yLow, xVal, yVal, seedValue):
    lowModel, scalerXLow, scalerYLow = BuildLowModel(config, xLow, yLow, seedValue)
    initialPoints = BuildInitialHighPoints(config, lowModel, scalerXLow, scalerYLow, xLow, seedValue + 10)
    initialCount = InitialHighCount(config)
    xHigh = np.asarray(initialPoints[:initialCount], dtype=float)
    yHigh = BoreholeHighFidelity(xHigh)

    fixedEta = ComputeFixedEta(xLow)
    bounds = BuildBoundsFromDataLocal(xLow)
    currentModelInfo = None
    prevModelInfo = None
    history = []

    while True:
        currentModelInfo = TrainFusionVariant(
            config,
            lowModel,
            scalerXLow,
            scalerYLow,
            xHigh,
            yHigh,
            seedValue + len(history) + 100,
            rhoMode="estimate",
        )
        metrics = EvaluateModel(currentModelInfo, scalerXLow, scalerYLow, xVal, yVal)
        history.append(
            {
                "method": "DART",
                "seed": seedValue,
                "current_high_count": len(xHigh),
                "mae": metrics["mae"],
            }
        )

        if len(xHigh) >= config.bhf:
            break

        newPoints = SelectPointsForGroup(
            "Baseline",
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

        remainingSlots = config.bhf - len(xHigh)
        newPoints = np.asarray(newPoints[:remainingSlots], dtype=float)
        xHigh = np.vstack([xHigh, newPoints])
        yHigh = np.hstack([yHigh, BoreholeHighFidelity(newPoints).ravel()])
        prevModelInfo = currentModelInfo

    historyFrame = pd.DataFrame(history)
    return {
        "history": historyFrame,
        "final_mae": float(historyFrame.iloc[-1]["mae"]),
    }


def RunGenericMethod(methodName, config, xLow, yLow, xVal, yVal, seedValue):
    lowModel, scalerXLow, scalerYLow = BuildLowModel(config, xLow, yLow, seedValue)
    xHigh = GenerateInitialHighPoints(config, xLow, seedValue + 5000)
    yHigh = BoreholeHighFidelity(xHigh)
    fixedEta = ComputeFixedEta(xLow)
    history = []

    while True:
        if methodName == "SF-Kriging":
            modelInfo = BuildSingleFidelityModel(config, xHigh, yHigh, seedValue + len(history) + 100)
            maeValue = EvaluateWithPredictor(PredictSingleFidelity, modelInfo, xVal, yVal)
            predictorFn = PredictSingleFidelity
        elif methodName == "Co-Kriging":
            modelInfo = BuildCoKrigingModel(
                config, lowModel, scalerXLow, scalerYLow, xHigh, yHigh, seedValue + len(history) + 100
            )
            maeValue = EvaluateWithPredictor(PredictCoKriging, modelInfo, xVal, yVal)
            predictorFn = PredictCoKriging
        else:
            modelInfo = BuildHierarchicalKrigingModel(
                config, lowModel, scalerXLow, scalerYLow, xHigh, yHigh, seedValue + len(history) + 100
            )
            maeValue = EvaluateWithPredictor(PredictHierarchicalKriging, modelInfo, xVal, yVal)
            predictorFn = PredictHierarchicalKriging

        history.append(
            {
                "method": methodName,
                "seed": seedValue,
                "current_high_count": len(xHigh),
                "mae": maeValue,
            }
        )

        if len(xHigh) >= config.bhf:
            break

        newPoint = SelectPointByStd(predictorFn, modelInfo, xLow, xHigh, fixedEta)
        if newPoint is None:
            break

        xHigh = np.vstack([xHigh, newPoint.reshape(1, -1)])
        yHigh = np.hstack([yHigh, BoreholeHighFidelity(newPoint.reshape(1, -1)).ravel()])

    historyFrame = pd.DataFrame(history)
    return {
        "history": historyFrame,
        "final_mae": float(historyFrame.iloc[-1]["mae"]),
    }


def RunSingleRepeat(seedValue):
    config = BuildBenchmarkConfig()
    dartConfig = BuildDartConfig(config)
    xLow, yLow = GenerateLowDataset(config, seedValue)
    xVal, yVal = GenerateValidationDataset(config)
    methodResults = {}

    dartResult = RunDartBaseline(dartConfig, xLow, yLow, xVal, yVal, seedValue)
    methodResults["DART"] = {
        "history": dartResult["history"][["current_high_count", "mae"]].copy(),
        "final_mae": dartResult["final_mae"],
    }

    for methodName in METHOD_ORDER:
        if methodName == "DART":
            continue
        result = RunGenericMethod(methodName, config, xLow, yLow, xVal, yVal, seedValue)
        methodResults[methodName] = {
            "history": result["history"][["current_high_count", "mae"]].copy(),
            "final_mae": result["final_mae"],
        }

    return {"seed": seedValue, "methods": methodResults}


def ConvertHistoryToCurve(historyFrame, initialCount, bhf):
    evalPoints = np.arange(initialCount, bhf + 1, dtype=int)
    curveFrame = historyFrame[["current_high_count", "mae"]].drop_duplicates(subset=["current_high_count"], keep="last")
    curveFrame = curveFrame.set_index("current_high_count").reindex(evalPoints).ffill()
    return curveFrame["mae"].to_numpy(dtype=float)


def AggregateCurves(historyList, initialCount, bhf):
    if not historyList:
        evalPoints = np.arange(initialCount, bhf + 1, dtype=int)
        nanArray = np.full(len(evalPoints), np.nan, dtype=float)
        return evalPoints, nanArray

    curves = [ConvertHistoryToCurve(historyFrame, initialCount, bhf) for historyFrame in historyList]
    curveMatrix = np.vstack(curves)
    evalPoints = np.arange(initialCount, bhf + 1, dtype=int)
    meanValues = np.nanmean(curveMatrix, axis=0)
    return evalPoints, meanValues


def RunThreadPool(seedValues):
    runResults = []
    maxWorkers = min(len(seedValues), max(1, int(os.environ.get("COMPARISON_MAX_WORKERS", "4"))))
    with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
        futureMap = {executor.submit(RunSingleRepeat, seedValue): seedValue for seedValue in seedValues}
        for future in as_completed(futureMap):
            runResults.append(future.result())
    return runResults


def RunProcessPool(seedValues):
    runResults = []
    maxWorkers = min(len(seedValues), max(1, int(os.environ.get("COMPARISON_MAX_WORKERS", "4"))))
    with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        futureMap = {executor.submit(RunSingleRepeat, seedValue): seedValue for seedValue in seedValues}
        for future in as_completed(futureMap):
            runResults.append(future.result())
    return runResults


def Main():
    config = BuildBenchmarkConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputDir = os.path.join(CONVERGENCE_DIR, f"borehole8_compare_{timestamp}")
    os.makedirs(outputDir, exist_ok=True)

    seedValues = [config.random_seed + runIndex * 100 for runIndex in range(config.repeat_count)]
    executorKind = os.environ.get("COMPARISON_EXECUTOR", "thread").strip().lower()

    try:
        if executorKind == "process":
            runResults = RunProcessPool(seedValues)
        else:
            runResults = RunThreadPool(seedValues)
    except (PermissionError, OSError, BrokenProcessPool):
        runResults = RunThreadPool(seedValues)

    runResults.sort(key=lambda item: item["seed"])
    initialCount = InitialHighCount(config)

    detailRows = []
    methodHistories = {methodName: [] for methodName in METHOD_ORDER}
    for runResult in runResults:
        for methodName in METHOD_ORDER:
            historyFrame = runResult["methods"][methodName]["history"].copy()
            historyFrame.insert(0, "benchmark", "borehole8")
            detailRows.append(historyFrame)
            methodHistories[methodName].append(runResult["methods"][methodName]["history"])

    detailFrame = pd.concat(detailRows, ignore_index=True)
    detailPath = os.path.join(outputDir, "borehole8_iteration_mae.csv")
    detailFrame.to_csv(detailPath, index=False, encoding="utf-8-sig")

    evalPoints, meanDart = AggregateCurves(methodHistories["DART"], initialCount, config.bhf)
    _, meanSf = AggregateCurves(methodHistories["SF-Kriging"], initialCount, config.bhf)
    _, meanCo = AggregateCurves(methodHistories["Co-Kriging"], initialCount, config.bhf)
    _, meanHk = AggregateCurves(methodHistories["Hierarchical Kriging"], initialCount, config.bhf)

    summaryFrame = pd.DataFrame(
        {
            "BHF": evalPoints,
            "DART_MeanMAE": meanDart,
            "SF-Kriging_MeanMAE": meanSf,
            "Co-Kriging_MeanMAE": meanCo,
            "HierarchicalKriging_MeanMAE": meanHk,
        }
    )
    summaryPath = os.path.join(outputDir, "borehole8_mean_curve.csv")
    summaryFrame.to_csv(summaryPath, index=False, encoding="utf-8-sig")

    metaPath = os.path.join(outputDir, "borehole8_run_config.json")
    with open(metaPath, "w", encoding="utf-8") as jsonFile:
        json.dump(
            {
                "BHF": config.bhf,
                "LOW_SAMPLE_COUNT": config.low_sample_count,
                "REPEAT_COUNT": config.repeat_count,
                "VALIDATION_COUNT": config.validation_count,
                "dimension": config.dimension,
            },
            jsonFile,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Result directory: {outputDir}")
    print(f"Per-iteration error data: {detailPath}")
    print(f"Summary mean curve: {summaryPath}")


if __name__ == "__main__":
    Main()
