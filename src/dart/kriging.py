"""
基于 NumPy 的轻量 Kriging 代理模型。
"""

import numpy as np

from .base import Ensure2D


def KernelRbf(distance):
    return np.exp(-0.5 * distance * distance)


def KernelMatern32(distance):
    sqrt3 = np.sqrt(3.0)
    scaledDistance = sqrt3 * distance
    return (1.0 + scaledDistance) * np.exp(-scaledDistance)


def KernelMatern52(distance):
    sqrt5 = np.sqrt(5.0)
    scaledDistance = sqrt5 * distance
    return (1.0 + scaledDistance + (scaledDistance * scaledDistance) / 3.0) * np.exp(-scaledDistance)


def PairwiseScaledDistances(xFirst, xSecond, lengthScale):
    diffMatrix = (xFirst[:, None, :] - xSecond[None, :, :]) / lengthScale
    return np.linalg.norm(diffMatrix, axis=2)


def ForwardSubstitution(lowerMatrix, rightSide):
    rightSide = np.asarray(rightSide, dtype=float)
    if rightSide.ndim == 1:
        rightSide = rightSide.reshape(-1, 1)

    dimension = lowerMatrix.shape[0]
    solution = np.zeros_like(rightSide, dtype=float)

    for rowIndex in range(dimension):
        solution[rowIndex] = (rightSide[rowIndex] - lowerMatrix[rowIndex, :rowIndex] @ solution[:rowIndex]) / lowerMatrix[rowIndex, rowIndex]

    return solution


def BackwardSubstitution(upperMatrix, rightSide):
    rightSide = np.asarray(rightSide, dtype=float)
    if rightSide.ndim == 1:
        rightSide = rightSide.reshape(-1, 1)

    dimension = upperMatrix.shape[0]
    solution = np.zeros_like(rightSide, dtype=float)

    for rowIndex in range(dimension - 1, -1, -1):
        solution[rowIndex] = (rightSide[rowIndex] - upperMatrix[rowIndex, rowIndex + 1:] @ solution[rowIndex + 1:]) / upperMatrix[rowIndex, rowIndex]

    return solution


def MatVec(matrixData, vectorData):
    matrixData = np.asarray(matrixData, dtype=float)
    vectorData = np.asarray(vectorData, dtype=float).ravel()
    outputVector = np.zeros(matrixData.shape[0], dtype=float)

    for rowIndex in range(matrixData.shape[0]):
        totalValue = 0.0
        for colIndex in range(matrixData.shape[1]):
            totalValue += matrixData[rowIndex, colIndex] * vectorData[colIndex]
        outputVector[rowIndex] = totalValue

    return outputVector


def SolveDenseSystem(matrixData, rightSide):
    matrixData = np.asarray(matrixData, dtype=float).copy()
    rightSide = np.asarray(rightSide, dtype=float).ravel().copy()
    dimension = matrixData.shape[0]

    for colIndex in range(dimension):
        pivotIndex = colIndex
        pivotValue = abs(matrixData[pivotIndex, colIndex])
        for rowIndex in range(colIndex + 1, dimension):
            currentValue = abs(matrixData[rowIndex, colIndex])
            if currentValue > pivotValue:
                pivotIndex = rowIndex
                pivotValue = currentValue

        if pivotValue < 1e-12:
            matrixData[colIndex, colIndex] += 1e-8
            pivotIndex = colIndex

        if pivotIndex != colIndex:
            matrixData[[colIndex, pivotIndex]] = matrixData[[pivotIndex, colIndex]]
            rightSide[[colIndex, pivotIndex]] = rightSide[[pivotIndex, colIndex]]

        diagonalValue = matrixData[colIndex, colIndex]
        if abs(diagonalValue) < 1e-12:
            diagonalValue = 1e-8
            matrixData[colIndex, colIndex] = diagonalValue

        for rowIndex in range(colIndex + 1, dimension):
            factor = matrixData[rowIndex, colIndex] / diagonalValue
            if factor == 0.0:
                continue
            matrixData[rowIndex, colIndex:] -= factor * matrixData[colIndex, colIndex:]
            rightSide[rowIndex] -= factor * rightSide[colIndex]

    solution = np.zeros(dimension, dtype=float)
    for rowIndex in range(dimension - 1, -1, -1):
        remainingValue = rightSide[rowIndex]
        for colIndex in range(rowIndex + 1, dimension):
            remainingValue -= matrixData[rowIndex, colIndex] * solution[colIndex]
        denominator = matrixData[rowIndex, rowIndex]
        if abs(denominator) < 1e-12:
            denominator = 1e-8
        solution[rowIndex] = remainingValue / denominator

    return solution


def FitTrendCoefficients(designMatrix, yData):
    designMatrix = np.asarray(designMatrix, dtype=float)
    yData = np.asarray(yData, dtype=float).ravel()
    featureCount = designMatrix.shape[1]

    if featureCount == 0:
        return np.empty((0,), dtype=float)

    gramMatrix = np.zeros((featureCount, featureCount), dtype=float)
    rightSide = np.zeros(featureCount, dtype=float)

    for rowIndex in range(designMatrix.shape[0]):
        rowValue = designMatrix[rowIndex]
        for firstIndex in range(featureCount):
            rightSide[firstIndex] += rowValue[firstIndex] * yData[rowIndex]
            for secondIndex in range(featureCount):
                gramMatrix[firstIndex, secondIndex] += rowValue[firstIndex] * rowValue[secondIndex]

    for diagIndex in range(featureCount):
        gramMatrix[diagIndex, diagIndex] += 1e-8

    return SolveDenseSystem(gramMatrix, rightSide)


def TrendMatrix(xData, polyType):
    xData = Ensure2D(xData).astype(float)
    trendType = (polyType or "constant").lower()

    if trendType in {"none", "zero"}:
        return np.empty((len(xData), 0), dtype=float)
    if trendType == "linear":
        return np.hstack([np.ones((len(xData), 1), dtype=float), xData])
    return np.ones((len(xData), 1), dtype=float)


class SimpleKrigingModel:
    def __init__(self, dim, corr="matern52", randomState=0, polyType="constant", nuggetVal=1e-8, lMin=0.5, lMax=20.0):
        self.dim = dim
        self.corr = (corr or "matern52").lower()
        self.randomState = randomState
        self.polyType = (polyType or "constant").lower()
        self.nuggetVal = float(max(nuggetVal, 1e-12))
        self.lMin = float(max(lMin, 1e-6))
        self.lMax = float(max(lMax, self.lMin))
        self.normalize = True
        self.xTrain = None
        self.yTrain = None
        self.lengthScale = np.ones(dim, dtype=float)
        self.yMean = 0.0
        self.yStd = 1.0
        self.trendBeta = None
        self.choleskyFactor = None
        self.alpha = None

    def Kernel(self, xFirst, xSecond):
        distance = PairwiseScaledDistances(xFirst, xSecond, self.lengthScale)
        if self.corr in {"squar_exp", "squared_exp", "gauss", "rbf"}:
            return KernelRbf(distance)
        if self.corr in {"matern32", "matern_32"}:
            return KernelMatern32(distance)
        return KernelMatern52(distance)

    def SetTrainingValues(self, xData, yData):
        self.xTrain = Ensure2D(xData).astype(float)
        self.yTrain = np.asarray(yData, dtype=float).ravel()

    def Train(self):
        if self.xTrain is None or self.yTrain is None:
            raise ValueError("训练前必须先设置训练数据。")

        xStd = np.std(self.xTrain, axis=0)
        xStd[xStd < 1e-6] = 1.0
        self.lengthScale = np.clip(xStd, self.lMin, self.lMax)

        yData = self.yTrain.copy()
        if self.normalize:
            self.yMean = float(np.mean(yData))
            self.yStd = float(np.std(yData))
            if self.yStd < 1e-12:
                self.yStd = 1.0
            yData = (yData - self.yMean) / self.yStd
        else:
            self.yMean = 0.0
            self.yStd = 1.0

        designMatrix = TrendMatrix(self.xTrain, self.polyType)
        if designMatrix.shape[1] > 0:
            self.trendBeta = FitTrendCoefficients(designMatrix, yData)
            residual = yData - MatVec(designMatrix, self.trendBeta)
        else:
            self.trendBeta = np.empty((0,), dtype=float)
            residual = yData

        covarianceMatrix = self.Kernel(self.xTrain, self.xTrain)
        covarianceMatrix[np.diag_indices_from(covarianceMatrix)] += self.nuggetVal

        try:
            self.choleskyFactor = np.linalg.cholesky(covarianceMatrix)
        except np.linalg.LinAlgError:
            covarianceMatrix[np.diag_indices_from(covarianceMatrix)] += max(self.nuggetVal, 1e-10) * 10.0
            self.choleskyFactor = np.linalg.cholesky(covarianceMatrix)

        tempValue = ForwardSubstitution(self.choleskyFactor, residual)
        self.alpha = BackwardSubstitution(self.choleskyFactor.T, tempValue).ravel()
        return self

    def PredictValues(self, xData):
        xData = Ensure2D(xData).astype(float)
        covarianceStar = self.Kernel(xData, self.xTrain)
        designStar = TrendMatrix(xData, self.polyType)

        trendValue = np.zeros(len(xData), dtype=float)
        if designStar.shape[1] > 0:
            trendValue = MatVec(designStar, self.trendBeta)

        meanNormalized = trendValue + covarianceStar @ self.alpha
        meanValue = meanNormalized * self.yStd + self.yMean
        return meanValue.reshape(-1, 1)

    def PredictVariances(self, xData):
        xData = Ensure2D(xData).astype(float)
        covarianceStar = self.Kernel(xData, self.xTrain)
        forwardValue = ForwardSubstitution(self.choleskyFactor, covarianceStar.T)
        varianceNormalized = np.maximum(1.0 - np.sum(forwardValue * forwardValue, axis=0), 1e-12)
        varianceValue = varianceNormalized * (self.yStd ** 2)
        return varianceValue.reshape(-1, 1)


def CreateKrigingModel(dim, theta0=None, randomState=0, polyType="constant", nuggetVal=1e-8, corr="matern52", lMin=0.5, lMax=20.0):
    return SimpleKrigingModel(
        dim=dim,
        corr=corr,
        randomState=randomState,
        polyType=polyType,
        nuggetVal=nuggetVal,
        lMin=lMin,
        lMax=lMax,
    )


def FitKriging(krgModel, xData, yData, normalize=True):
    krgModel.SetTrainingValues(xData, yData)
    krgModel.normalize = normalize
    krgModel.Train()
    return krgModel


def PredictKriging(krgModel, xData, returnStd=False):
    xData = Ensure2D(xData)
    if returnStd:
        yPred = krgModel.PredictValues(xData)
        yVar = krgModel.PredictVariances(xData)
        return yPred.flatten(), np.sqrt(yVar.flatten())
    return krgModel.PredictValues(xData).flatten()


# Compatibility aliases for legacy scripts.
create_kriging_model = CreateKrigingModel
fit_kriging = FitKriging
predict_kriging = PredictKriging
