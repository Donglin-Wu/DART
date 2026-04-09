"""
序贯融合主流程。
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import CreateResultDirectory, ReadCsvAuto, logger
from .fusion import FusionModel
from .initialization import BuildBoundsFromData, LatinHypercubeSampling, SelectInitialPoints
from .kriging import CreateKrigingModel, FitKriging, PredictKriging
from .optimizer import PsoSearchExtrema
from .predict_and_plot_from_pkl import GeneratePredictionAndScatter
from .strategy import SelectNewPoints
from .visualization import VisualizeResults


POINT_TOL = 1e-6


def DefaultModelConfig(krgCorr="squar_exp", overrides=None):
    config = {
        "low": {
            "corr": krgCorr,
            "poly_type": "constant",
            "nugget_val": 1e-3,
            "l_min": 0.25,
            "l_max": 10.0,
        },
        "delta": {
            "corr": krgCorr,
            "poly_type": "linear",
            "nugget_val": 1e-8,
            "l_min": 0.25,
            "l_max": 10.0,
        },
    }

    if overrides:
        for scopeName in ["low", "delta"]:
            if scopeName in overrides and overrides[scopeName]:
                config[scopeName].update(overrides[scopeName])

    return config


def To2DArray(points, dim):
    if points is None:
        return np.empty((0, dim), dtype=float)
    arrayData = np.asarray(points, dtype=float)
    if arrayData.size == 0:
        return np.empty((0, dim), dtype=float)
    return arrayData.reshape(-1, dim)


def StackPoints(dim, *pointSets):
    validArrays = []
    for pointSet in pointSets:
        pointArray = To2DArray(pointSet, dim)
        if len(pointArray) > 0:
            validArrays.append(pointArray)

    if not validArrays:
        return np.empty((0, dim), dtype=float)

    return np.vstack(validArrays)


def PointExists(points, point, tol=POINT_TOL):
    point = np.asarray(point, dtype=float)
    pointArray = To2DArray(points, point.shape[0])
    if len(pointArray) == 0:
        return False
    return bool(np.any(np.linalg.norm(pointArray - point, axis=1) <= tol))


def SnapPointToLowFidelity(targetPoint, xLow, usedPoints=None):
    targetPoint = np.asarray(targetPoint, dtype=float)
    dim = xLow.shape[1]
    usedArray = To2DArray(usedPoints, dim)
    distances = np.linalg.norm(xLow - targetPoint, axis=1)

    for candidateIndex in np.argsort(distances):
        candidatePoint = xLow[candidateIndex]
        if PointExists(usedArray, candidatePoint):
            continue
        return candidatePoint.copy()

    return None


def SnapTargetsToLowFidelity(targetPoints, xLow, usedPoints=None):
    dim = xLow.shape[1]
    snappedPoints = []

    for targetPoint in targetPoints:
        if targetPoint is None:
            snappedPoints.append(None)
            continue

        mergedUsedPoints = StackPoints(dim, usedPoints, [point for point in snappedPoints if point is not None])
        snappedPoint = SnapPointToLowFidelity(targetPoint, xLow, usedPoints=mergedUsedPoints)
        snappedPoints.append(snappedPoint)

    return snappedPoints


def FillWithLowFidelityPoints(selectedPoints, xLow, targetCount):
    dim = xLow.shape[1]
    selectedArray = To2DArray(selectedPoints, dim)

    while len(selectedArray) < targetCount:
        bestCandidate = None
        bestScore = -np.inf

        for candidatePoint in xLow:
            if PointExists(selectedArray, candidatePoint):
                continue

            if len(selectedArray) == 0:
                currentScore = 0.0
            else:
                currentScore = float(np.min(np.linalg.norm(selectedArray - candidatePoint, axis=1)))

            if currentScore > bestScore:
                bestScore = currentScore
                bestCandidate = candidatePoint

        if bestCandidate is None:
            break

        selectedArray = StackPoints(dim, selectedArray, [bestCandidate])

    return selectedArray


def ReorderSamplesToReference(xSamples, ySamples, referencePoints, tol=POINT_TOL):
    dim = referencePoints.shape[1]
    xSamples = To2DArray(xSamples, dim)
    ySamples = np.asarray(ySamples, dtype=float).ravel()

    if len(xSamples) != len(referencePoints):
        raise ValueError(f"高精度样本数量必须与推荐点数量一致，期望 {len(referencePoints)}，实际 {len(xSamples)}。")

    remainingIndexes = list(range(len(xSamples)))
    orderedX = []
    orderedY = []

    for referencePoint in referencePoints:
        bestIndex = None
        bestDistance = np.inf

        for sampleIndex in remainingIndexes:
            currentDistance = float(np.linalg.norm(xSamples[sampleIndex] - referencePoint))
            if currentDistance < bestDistance:
                bestDistance = currentDistance
                bestIndex = sampleIndex

        if bestIndex is None or bestDistance > tol:
            raise ValueError(f"高精度样本坐标与推荐点不一致，最近距离为 {bestDistance:.3e}。")

        orderedX.append(xSamples[bestIndex])
        orderedY.append(ySamples[bestIndex])
        remainingIndexes.remove(bestIndex)

    return np.asarray(orderedX), np.asarray(orderedY)


def EstimateRho(yHigh, yLowPred, defaultValue=1.0):
    yHigh = np.asarray(yHigh, dtype=float).ravel()
    yLowPred = np.asarray(yLowPred, dtype=float).ravel()

    denominator = float(np.dot(yLowPred, yLowPred))
    if denominator <= 1e-12:
        return float(defaultValue)

    rhoValue = float(np.dot(yHigh, yLowPred) / denominator)
    if not np.isfinite(rhoValue):
        return float(defaultValue)
    return rhoValue


def InitialN(dim, fes):
    initialCount = 5 * dim
    initialCount = min(initialCount, fes)
    initialCount = max(4, initialCount)
    logger.info(f"按维度估计初始高精度样本数: d={dim}, N={initialCount}")
    return initialCount


def BuildGuiState(xLow, yLow, xHigh, yHigh, fusionModel, scalerXLow, scalerYLow, bounds, dim, fes, iterations, inputColumns, outputColumn, xVal=None, yVal=None):
    return {
        "xLow": np.array(xLow, dtype=float, copy=True),
        "yLow": np.array(yLow, dtype=float, copy=True),
        "xHigh": np.array(xHigh, dtype=float, copy=True),
        "yHigh": np.array(yHigh, dtype=float, copy=True),
        "fusionModel": fusionModel,
        "scalerXLow": scalerXLow,
        "scalerYLow": scalerYLow,
        "bounds": list(bounds),
        "dim": dim,
        "fes": fes,
        "iterations": iterations,
        "inputColumns": list(inputColumns),
        "outputColumn": outputColumn,
        "xVal": None if xVal is None else np.array(xVal, dtype=float, copy=True),
        "yVal": None if yVal is None else np.array(yVal, dtype=float, copy=True),
    }


def EmitGuiState(interactor, stateData):
    if interactor is not None and hasattr(interactor, "UpdateState"):
        interactor.UpdateState(stateData)


def GetInput(points, dim, isInitial=False, xHigh=None, yHigh=None, interactor=None):
    xHigh = xHigh if xHigh is not None else []
    yHigh = yHigh if yHigh is not None else []

    xNew = []
    yNew = []
    validPoints = [np.asarray(point, dtype=float) for point in points if point is not None]

    if len(validPoints) == 0:
        logger.warning("当前没有可填写的推荐点。")
        return np.empty((0, dim)), np.empty((0,), dtype=float)

    pointType = "初始高精度推荐点" if isInitial else "新增高精度推荐点"

    if interactor is not None and hasattr(interactor, "RequestPointValues"):
        while True:
            responseList, stopRequested = interactor.RequestPointValues(validPoints, dim, isInitial, xHigh, yHigh)
            if stopRequested:
                logger.info("用户请求提前结束，停止后续高精度点输入。")
                return np.empty((0, dim)), np.empty((0,), dtype=float)

            try:
                for pointIndex, point in enumerate(validPoints):
                    responseText = ""
                    if pointIndex < len(responseList):
                        responseText = (responseList[pointIndex] or "").strip()

                    if responseText.lower() == "skip":
                        logger.info(f"跳过第 {pointIndex + 1} 个推荐点。")
                        continue

                    valueList = [float(item) for item in responseText.split(",") if item.strip() != ""]
                    if len(valueList) == 1:
                        xValue = point.copy()
                        yValue = float(valueList[0])
                    elif len(valueList) == dim + 1:
                        xValue = np.array(valueList[:-1], dtype=float)
                        if np.linalg.norm(xValue - point) > POINT_TOL:
                            raise ValueError(f"第 {pointIndex + 1} 个输入坐标必须与推荐点完全一致: {point}")
                        yValue = float(valueList[-1])
                    else:
                        raise ValueError(f"第 {pointIndex + 1} 个输入格式错误，应为 1 个值或 {dim + 1} 个逗号分隔值。")

                    duplicatePoint = False
                    existingArray = To2DArray(xHigh, dim)
                    existingY = np.asarray(yHigh).ravel()

                    for historyIndex, historyPoint in enumerate(existingArray):
                        if np.linalg.norm(xValue - historyPoint) <= POINT_TOL:
                            raise ValueError(
                                f"第 {pointIndex + 1} 个输入坐标重复，历史点为第 {historyIndex + 1} 个: {historyPoint}，历史高精度值为 {existingY[historyIndex]}"
                            )

                    if not duplicatePoint and PointExists(xNew, xValue):
                        raise ValueError(f"第 {pointIndex + 1} 个输入坐标在当前批次中重复。")

                    xNew.append(xValue)
                    yNew.append(yValue)
                    logger.info(f"已记录第 {pointIndex + 1} 个点: x={xValue}, y={yValue}")

                return To2DArray(xNew, dim), np.asarray(yNew, dtype=float)
            except Exception as exc:
                xNew = []
                yNew = []
                logger.warning(f"界面输入校验失败: {exc}")
                if hasattr(interactor, "NotifyInputError"):
                    interactor.NotifyInputError(str(exc))

    for pointIndex, point in enumerate(validPoints):
        print(f"\n{pointType} {pointIndex + 1}/{len(validPoints)}: {point}")

        while True:
            inputText = input("请输入该点的高精度 y 值，或输入 x1,...,xd,y；输入 skip 可跳过: ").strip()

            if inputText.lower() == "skip":
                logger.info(f"跳过第 {pointIndex + 1} 个推荐点。")
                break

            try:
                valueList = [float(item) for item in inputText.split(",") if item.strip() != ""]

                if len(valueList) == 1:
                    xValue = point.copy()
                    yValue = float(valueList[0])
                elif len(valueList) == dim + 1:
                    xValue = np.array(valueList[:-1], dtype=float)
                    if np.linalg.norm(xValue - point) > POINT_TOL:
                        print(f"输入坐标必须与推荐点完全一致: {point}")
                        continue
                    yValue = float(valueList[-1])
                else:
                    print(f"输入格式错误，应为 1 个值，或 {dim + 1} 个逗号分隔的值。")
                    continue

                duplicatePoint = False
                existingArray = To2DArray(xHigh, dim)
                existingY = np.asarray(yHigh).ravel()

                for historyIndex, historyPoint in enumerate(existingArray):
                    if np.linalg.norm(xValue - historyPoint) <= POINT_TOL:
                        print(f"该坐标已存在，对应历史点为第 {historyIndex + 1} 个: {historyPoint}")
                        print(f"历史高精度值为: {existingY[historyIndex]}")
                        duplicatePoint = True
                        break

                if not duplicatePoint and PointExists(xNew, xValue):
                    print("该坐标在本轮输入中重复。")
                    duplicatePoint = True

                if duplicatePoint:
                    continue

                xNew.append(xValue)
                yNew.append(yValue)
                logger.info(f"已记录第 {pointIndex + 1} 个点: x={xValue}, y={yValue}")
                break

            except Exception as exc:
                print(f"输入解析失败: {exc}")

    return To2DArray(xNew, dim), np.asarray(yNew, dtype=float)


def TrainFusionModel(krgLow, xHigh, yHigh, scalerXLow, scalerYLow, dim, randomState, deltaConfig):
    xHighLowScaled = scalerXLow.transform(xHigh)
    yLowPredScaled = PredictKriging(krgLow, xHighLowScaled, returnStd=False)
    yLowPred = scalerYLow.inverse_transform(yLowPredScaled.reshape(-1, 1)).ravel()

    rhoValue = EstimateRho(yHigh, yLowPred)
    yDelta = yHigh - rhoValue * yLowPred

    scalerXHigh = StandardScaler().fit(xHigh)
    scalerYDelta = StandardScaler().fit(yDelta.reshape(-1, 1))

    xHighScaled = scalerXHigh.transform(xHigh)
    yDeltaScaled = scalerYDelta.transform(yDelta.reshape(-1, 1)).ravel()

    krgDelta = CreateKrigingModel(
        dim=dim,
        randomState=randomState,
        polyType=deltaConfig["poly_type"],
        nuggetVal=deltaConfig["nugget_val"],
        corr=deltaConfig["corr"],
        lMin=deltaConfig["l_min"],
        lMax=deltaConfig["l_max"],
    )
    krgDelta = FitKriging(krgDelta, xHighScaled, yDeltaScaled, normalize=True)

    fusionModel = FusionModel(krgLow, krgDelta, rhoValue, scalerXHigh, scalerYDelta)
    return fusionModel, rhoValue


def SequentialFusion(cfdCsv, fes=30, maxIters=50, randomState=42, krgCorr="squar_exp", validationCsv=None, modelConfig=None, outputBaseDir=None, interactor=None):
    resultDir, timestamp = CreateResultDirectory(outputBaseDir)
    logger.info(f"本次结果目录: {resultDir}")
    logger.info(f"低精度数据文件: {cfdCsv}")
    logger.info(f"高精度预算 Fes: {fes}")

    modelConfig = DefaultModelConfig(krgCorr=krgCorr, overrides=modelConfig)

    logger.info("步骤 1：读取低精度数据。")
    xLow, yLow, inputColumns, outputColumn = ReadCsvAuto(cfdCsv)
    dim = xLow.shape[1]
    logger.info(f"低精度样本数: {len(xLow)}")
    logger.info(f"输入维度: {dim}")

    xVal = None
    yVal = None
    if validationCsv and os.path.exists(validationCsv):
        try:
            xVal, yVal, _, _ = ReadCsvAuto(validationCsv)
            logger.info(f"已加载验证数据，共 {len(xVal)} 个样本。")
        except Exception as exc:
            logger.warning(f"验证数据读取失败: {exc}")

    if len(xLow) > 1:
        diffMatrix = xLow[:, np.newaxis, :] - xLow[np.newaxis, :, :]
        distanceMatrix = np.linalg.norm(diffMatrix, axis=-1)
        np.fill_diagonal(distanceMatrix, np.inf)
        fixedEta = float(np.min(distanceMatrix))
        if fixedEta < 1e-6:
            fixedEta = 0.01
    else:
        fixedEta = 0.1
    logger.info(f"最小采样间距阈值 eta: {fixedEta:.6f}")

    initialCount = InitialN(dim, fes)

    scalerXLow = StandardScaler().fit(xLow)
    scalerYLow = StandardScaler().fit(yLow.reshape(-1, 1))
    xLowScaled = scalerXLow.transform(xLow)
    yLowScaled = scalerYLow.transform(yLow.reshape(-1, 1)).ravel()

    logger.info("步骤 2：训练低精度 Kriging 模型。")
    lowConfig = modelConfig["low"]
    krgLow = CreateKrigingModel(
        dim=dim,
        randomState=randomState,
        polyType=lowConfig["poly_type"],
        nuggetVal=lowConfig["nugget_val"],
        corr=lowConfig["corr"],
        lMin=lowConfig["l_min"],
        lMax=lowConfig["l_max"],
    )
    krgLow = FitKriging(krgLow, xLowScaled, yLowScaled, normalize=True)
    logger.info("低精度 Kriging 模型训练完成。")

    logger.info("步骤 3：在低精度模型上搜索极值点。")
    bounds = BuildBoundsFromData(xLow)

    def LowFidelityFunc(point):
        pointScaled = scalerXLow.transform(np.asarray(point, dtype=float).reshape(1, -1))
        predScaled = PredictKriging(krgLow, pointScaled)
        predValue = scalerYLow.inverse_transform(predScaled.reshape(-1, 1)).ravel()
        return float(predValue[0])

    searchCount = 2 * dim
    minPoints, maxPoints = PsoSearchExtrema(
        modelFunc=LowFidelityFunc,
        bounds=bounds,
        searchCount=searchCount,
        popSize=5 * dim,
        iters=30,
        randomState=randomState,
    )

    logger.info("步骤 4：构造初始连续推荐点。")
    selectedExtrema = SelectInitialPoints(minPoints, maxPoints)
    currentPointList = selectedExtrema.tolist() if len(selectedExtrema) > 0 else []

    for dimIndex in range(dim):
        lowerBound = bounds[dimIndex][0]
        upperBound = bounds[dimIndex][1]
        spanValue = upperBound - lowerBound
        toleranceValue = 0.05 * spanValue

        hasLowerBound = False
        if len(currentPointList) > 0:
            existingArray = np.array(currentPointList)
            if np.min(np.abs(existingArray[:, dimIndex] - lowerBound)) < toleranceValue:
                hasLowerBound = True

        if not hasLowerBound:
            newPoint = np.array([(bound[0] + bound[1]) / 2.0 for bound in bounds], dtype=float)
            newPoint[dimIndex] = lowerBound
            currentPointList.append(newPoint)
            logger.info(f"补充下边界覆盖点: {newPoint}")

        hasUpperBound = False
        if len(currentPointList) > 0:
            existingArray = np.array(currentPointList)
            if np.min(np.abs(existingArray[:, dimIndex] - upperBound)) < toleranceValue:
                hasUpperBound = True

        if not hasUpperBound:
            newPoint = np.array([(bound[0] + bound[1]) / 2.0 for bound in bounds], dtype=float)
            newPoint[dimIndex] = upperBound
            currentPointList.append(newPoint)
            logger.info(f"补充上边界覆盖点: {newPoint}")

    initialTargets = To2DArray(currentPointList, dim)
    logger.info(f"低精度映射前的连续推荐点数量: {len(initialTargets)}")

    logger.info(f"步骤 5：使用拉丁超立方补齐到 N={initialCount}。")
    rng = np.random.RandomState(randomState + 500)
    if len(initialTargets) < initialCount:
        neededCount = initialCount - len(initialTargets)
        candidateCount = max(neededCount * 100, 1000)
        candidatePool = LatinHypercubeSampling(bounds, candidateCount, randomState=randomState + 300)

        for _ in range(neededCount):
            if len(initialTargets) == 0:
                bestCandidate = candidatePool[0]
            else:
                bestCandidate = None
                bestDistance = -1.0
                shuffledIndexes = rng.permutation(len(candidatePool))[:500]
                for candidateIndex in shuffledIndexes:
                    candidatePoint = candidatePool[candidateIndex]
                    minDistance = float(np.min(np.linalg.norm(initialTargets - candidatePoint, axis=1)))
                    if minDistance > bestDistance:
                        bestDistance = minDistance
                        bestCandidate = candidatePoint

            initialTargets = np.vstack([initialTargets, bestCandidate])

        logger.info(f"已补充 {neededCount} 个连续推荐点。")
    else:
        logger.info("当前连续推荐点数量已达到初始要求。")

    logger.info("步骤 6：将推荐点映射到低精度数据中已有坐标。")
    snappedInitialPoints = [point for point in SnapTargetsToLowFidelity(initialTargets, xLow) if point is not None]
    initialPoints = FillWithLowFidelityPoints(snappedInitialPoints, xLow, initialCount)

    if len(initialPoints) < initialCount:
        logger.warning(f"可用低精度坐标不足，初始推荐点数量从 {initialCount} 降到 {len(initialPoints)}。")

    print(f"初始高精度推荐点数量: {len(initialPoints)}")
    print("以下推荐点均来自低精度数据中已有的坐标:")
    for pointIndex, point in enumerate(initialPoints):
        print(f"点 {pointIndex + 1}: {point}")

    if interactor is not None and hasattr(interactor, "ShowInitialPoints"):
        interactor.ShowInitialPoints(initialPoints)

    while True:
        if interactor is not None and hasattr(interactor, "RequestInitialHighCsv"):
            highCsvPath = interactor.RequestInitialHighCsv(initialPoints)
            if interactor is not None and hasattr(interactor, "ShouldStop") and interactor.ShouldStop():
                logger.info("用户已提前结束，停止融合流程。")
                break
        else:
            highCsvPath = input("\n请输入初始高精度 CSV 文件路径: ").strip()

        if not highCsvPath:
            print("初始高精度 CSV 路径不能为空。")
            continue

        if not os.path.exists(highCsvPath):
            print(f"文件不存在: {highCsvPath}")
            continue

        try:
            highFrame = pd.read_csv(highCsvPath)
            if highFrame.shape[1] < dim + 1:
                print(f"CSV 格式错误，至少需要 {dim} 列输入和 1 列输出。")
                continue

            xHigh = highFrame.iloc[:, :dim].values.astype(float)
            yHigh = highFrame.iloc[:, -1].values.astype(float)

            if np.any(np.isnan(xHigh)) or np.any(np.isinf(xHigh)):
                print("检测到高精度输入中存在 NaN 或 Inf，程序将自动转换。")
                xHigh = np.nan_to_num(xHigh)

            if np.any(np.isnan(yHigh)) or np.any(np.isinf(yHigh)):
                print("检测到高精度输出中存在 NaN 或 Inf，程序将自动转换。")
                yHigh = np.nan_to_num(yHigh)

            xHigh, yHigh = ReorderSamplesToReference(xHigh, yHigh, initialPoints)
            logger.info(f"已读取并校验初始高精度样本，共 {len(xHigh)} 个。")
            EmitGuiState(
                interactor,
                BuildGuiState(
                    xLow=xLow,
                    yLow=yLow,
                    xHigh=xHigh,
                    yHigh=yHigh,
                    fusionModel=None,
                    scalerXLow=scalerXLow,
                    scalerYLow=scalerYLow,
                    bounds=bounds,
                    dim=dim,
                    fes=fes,
                    iterations=0,
                    inputColumns=inputColumns,
                    outputColumn=outputColumn,
                    xVal=xVal,
                    yVal=yVal,
                ),
            )
            break
        except Exception as exc:
            print(f"初始高精度 CSV 读取或校验失败: {exc}")

    if interactor is not None and hasattr(interactor, "ShouldStop") and interactor.ShouldStop():
        xHigh = To2DArray([], dim)
        yHigh = np.asarray([], dtype=float)

    logger.info("步骤 7：开始序贯融合迭代。")
    logRecords = []
    prevFusionModel = None
    finalFusionModel = None
    finalRho = None

    for iterIndex in range(1, maxIters + 1):
        if interactor is not None and hasattr(interactor, "ShouldStop") and interactor.ShouldStop():
            logger.info("检测到用户提前结束请求，停止后续迭代。")
            break

        currentHighCount = len(xHigh)
        logger.info("\n" + "=" * 60)
        logger.info(f"第 {iterIndex} 轮迭代")
        logger.info("=" * 60)
        logger.info(f"当前高精度样本数: {currentHighCount}/{fes}")

        currentFusionModel, rhoValue = TrainFusionModel(
            krgLow=krgLow,
            xHigh=xHigh,
            yHigh=yHigh,
            scalerXLow=scalerXLow,
            scalerYLow=scalerYLow,
            dim=dim,
            randomState=randomState + iterIndex,
            deltaConfig=modelConfig["delta"],
        )
        finalFusionModel = currentFusionModel
        finalRho = rhoValue
        logger.info(f"当前估计 rho = {rhoValue:.6f}")
        EmitGuiState(
            interactor,
            BuildGuiState(
                xLow=xLow,
                yLow=yLow,
                xHigh=xHigh,
                yHigh=yHigh,
                fusionModel=currentFusionModel,
                scalerXLow=scalerXLow,
                scalerYLow=scalerYLow,
                bounds=bounds,
                dim=dim,
                fes=fes,
                iterations=iterIndex,
                inputColumns=inputColumns,
                outputColumn=outputColumn,
                xVal=xVal,
                yVal=yVal,
            ),
        )

        if currentHighCount >= fes:
            logger.info(f"已达到高精度预算 Fes={fes}，停止迭代。")
            break

        if interactor is not None and hasattr(interactor, "ShouldStop") and interactor.ShouldStop():
            logger.info("检测到用户提前结束请求，停止后续迭代。")
            break

        logger.info("步骤 8：计算新增推荐点并映射到低精度已有坐标。")
        rawPointDiff, rawPointUncertainty = SelectNewPoints(
            prevHfModel=prevFusionModel,
            currentHfModel=currentFusionModel,
            bounds=bounds,
            scalerXLow=scalerXLow,
            scalerYLow=scalerYLow,
            xHigh=xHigh,
            fixedEta=fixedEta,
            randomState=randomState + 1000 * iterIndex,
        )

        pointDiff, pointUncertainty = SnapTargetsToLowFidelity([rawPointDiff, rawPointUncertainty], xLow, usedPoints=xHigh)
        pointsToGet = [point for point in [pointDiff, pointUncertainty] if point is not None]

        if len(pointsToGet) == 0:
            logger.info("没有可用于新增高精度采样的低精度已有坐标，停止迭代。")
            break

        logger.info("本轮推荐点如下:")
        if pointDiff is not None:
            logger.info(f"1. 偏利用型推荐点: {pointDiff}")
        if pointUncertainty is not None:
            logger.info(f"2. 偏探索型推荐点: {pointUncertainty}")

        print("\n请根据以下推荐点补充高精度值。")
        if interactor is not None and hasattr(interactor, "ShowRecommendedPoints"):
            interactor.ShowRecommendedPoints(pointsToGet)

        xNew, yNew = GetInput(pointsToGet, dim, isInitial=False, xHigh=xHigh, yHigh=yHigh, interactor=interactor)

        if len(xNew) > 0:
            xHigh = np.vstack([xHigh, xNew])
            yHigh = np.hstack([yHigh, yNew])
            logger.info(f"本轮新增 {len(xNew)} 个高精度样本，当前总数 {len(xHigh)}。")

        logRecords.append(
            {
                "iteration": iterIndex,
                "current_high_count": len(xHigh),
                "rho": rhoValue,
                "point_diff": pointDiff.tolist() if pointDiff is not None else None,
                "point_uncertainty": pointUncertainty.tolist() if pointUncertainty is not None else None,
                "requested_point_count": len(pointsToGet),
                "actual_y_diff": float(yNew[0]) if len(yNew) > 0 else None,
                "actual_y_uncertainty": float(yNew[1]) if len(yNew) > 1 else None,
            }
        )
        prevFusionModel = currentFusionModel

    logger.info("步骤 9：输出结果文件。")

    usedHighPath = os.path.join(resultDir, "used_high_fidelity_data.csv")
    usedHighFrame = pd.DataFrame(np.hstack([xHigh, yHigh.reshape(-1, 1)]))
    inputNameList = [f"x{i + 1}" for i in range(dim)]
    usedHighFrame.columns = inputNameList + ["y_high"]
    usedHighFrame.to_csv(usedHighPath, index=False)

    visualizationPath = None

    if finalFusionModel is None and len(xHigh) > 0:
        finalFusionModel, finalRho = TrainFusionModel(
            krgLow=krgLow,
            xHigh=xHigh,
            yHigh=yHigh,
            scalerXLow=scalerXLow,
            scalerYLow=scalerYLow,
            dim=dim,
            randomState=randomState + max(len(logRecords), 1),
            deltaConfig=modelConfig["delta"],
        )

    if dim in (1, 2) and finalFusionModel is not None:
        visualizationPath = VisualizeResults(
            xLow=xLow,
            yLow=yLow,
            xHigh=xHigh,
            yHigh=yHigh,
            fusionModel=finalFusionModel,
            scalerXLow=scalerXLow,
            scalerYLow=scalerYLow,
            bounds=bounds,
            dim=dim,
            iterations=len(logRecords),
            fes=fes,
            outputDir=resultDir,
            xVal=xVal,
            yVal=yVal,
            inputLabels=inputColumns,
            outputLabel=outputColumn,
        )

    logDataFrame = pd.DataFrame(logRecords)
    logPath = os.path.join(resultDir, "fusion_log.csv")
    logDataFrame.to_csv(logPath, index=False)

    finalParams = {
        "n_low_fidelity": len(xLow),
        "n_high_fidelity": len(xHigh),
        "total_iterations": len(logRecords),
        "final_high_count": len(xHigh),
        "Fes": fes,
        "dimension": dim,
        "initial_high_count": len(initialPoints),
        "krg_correlation": krgCorr,
        "rho": finalRho,
        "model_config": modelConfig,
        "result_dir": resultDir,
        "timestamp": timestamp,
    }
    parameterPath = os.path.join(resultDir, "fusion_parameters.json")
    with open(parameterPath, "w", encoding="utf-8") as parameterFile:
        json.dump(finalParams, parameterFile, indent=2, ensure_ascii=False)

    modelPath = None
    predictionInfo = {"predictionCsv": None, "scatterSvg": None}
    if finalFusionModel is not None:
        modelData = {
            "fusion_model": finalFusionModel,
            "scaler_X_low": scalerXLow,
            "scaler_y_low": scalerYLow,
            "dim": dim,
            "model_config": modelConfig,
            "result_dir": resultDir,
        }
        modelPath = os.path.join(resultDir, "trained_model.pkl")
        with open(modelPath, "wb") as modelFile:
            pickle.dump(modelData, modelFile)

        predictionPath = os.path.join(resultDir, "fusion_predictions.csv")
        scatterPath = os.path.join(resultDir, "regression_scatter.svg")
        predictionInfo = GeneratePredictionAndScatter(
            modelPath=modelPath,
            inputCsvPath=cfdCsv,
            outputCsvPath=predictionPath,
            truthCsvPath=validationCsv if validationCsv and os.path.exists(validationCsv) else None,
            scatterSvgPath=scatterPath,
        )

    logger.info(f"已保存高精度样本数据: {usedHighPath}")
    logger.info(f"已保存运行日志: {logPath}")
    logger.info(f"已保存模型参数: {parameterPath}")
    if modelPath is not None:
        logger.info(f"已保存模型文件: {modelPath}")
    if visualizationPath:
        logger.info(f"已保存融合可视化结果: {visualizationPath}")
    if predictionInfo["predictionCsv"] is not None:
        logger.info(f"已保存融合预测结果: {predictionInfo['predictionCsv']}")
    if predictionInfo["scatterSvg"] is not None:
        logger.info(f"已保存回归拟合散点图: {predictionInfo['scatterSvg']}")

    return {
        "fusionModel": finalFusionModel,
        "xHigh": xHigh,
        "yHigh": yHigh,
        "logDataFrame": logDataFrame,
        "resultDir": resultDir,
        "modelPath": modelPath,
        "parameterPath": parameterPath,
        "logPath": logPath,
        "usedHighPath": usedHighPath,
        "visualizationPath": visualizationPath,
        "predictionCsvPath": predictionInfo["predictionCsv"],
        "scatterSvgPath": predictionInfo["scatterSvg"],
    }


# Compatibility aliases for legacy scripts.
to_2d_array = To2DArray
snap_targets_to_low_fidelity = SnapTargetsToLowFidelity
fill_with_low_fidelity_points = FillWithLowFidelityPoints
initial_n = InitialN
get_input = GetInput
sequential_fusion = SequentialFusion
latin_hypercube_sampling = LatinHypercubeSampling
visualize_results = VisualizeResults
