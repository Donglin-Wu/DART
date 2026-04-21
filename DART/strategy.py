"""
Sequential point-selection strategy.
"""

import numpy as np

from kriging import PredictKriging
from optimizer import PSO


def SelectNewPoints(prevHfModel, currentHfModel, bounds, scalerXLow, scalerYLow, xHigh=None, fixedEta=0.1, randomState=None):
    """
    Select new points using two objectives:
    1. Point with the largest model change
    2. Point with the largest uncertainty
    """
    dimension = len(bounds)
    popSize = 5 * dimension

    if prevHfModel is not None:
        def DiffObjective(point):
            if xHigh is not None and len(xHigh) > 0:
                distances = np.linalg.norm(xHigh - point, axis=1)
                if np.min(distances) <= (fixedEta / 2):
                    return -1e10

            prevMean = prevHfModel.Predict(point, scalerXLow=scalerXLow, scalerYLow=scalerYLow)
            currentMean = currentHfModel.Predict(point, scalerXLow=scalerXLow, scalerYLow=scalerYLow)
            return abs(float(currentMean[0]) - float(prevMean[0]))

        diffPso = PSO(
            objFunc=DiffObjective,
            bounds=bounds,
            popSize=popSize,
            iters=30,
            randomState=randomState,
        )
        pointDiff, _ = diffPso.Optimize()
    else:
        def FirstDiffObjective(point):
            if xHigh is not None and len(xHigh) > 0:
                distances = np.linalg.norm(xHigh - point, axis=1)
                if np.min(distances) <= (fixedEta / 2):
                    return -1e10

            currentMean = currentHfModel.Predict(point, scalerXLow=scalerXLow, scalerYLow=scalerYLow)
            xLowScaled = scalerXLow.transform(np.asarray(point, dtype=float).reshape(1, -1))
            lowMeanScaled = PredictKriging(currentHfModel.krgLow, xLowScaled, returnStd=False)
            lowMean = scalerYLow.inverse_transform(lowMeanScaled.reshape(-1, 1)).ravel()
            return abs(float(currentMean[0]) - currentHfModel.rho * float(lowMean[0]))

        firstDiffPso = PSO(
            objFunc=FirstDiffObjective,
            bounds=bounds,
            popSize=popSize,
            iters=30,
            randomState=randomState,
        )
        pointDiff, _ = firstDiffPso.Optimize()

    def UncertaintyObjective(point):
        if xHigh is not None and len(xHigh) > 0:
            distances = np.linalg.norm(xHigh - point, axis=1)
            if np.min(distances) < (fixedEta / 2):
                return -1e10

        _, stdValue = currentHfModel.Predict(
            point,
            returnStd=True,
            scalerXLow=scalerXLow,
            scalerYLow=scalerYLow,
        )
        return float(stdValue[0])

    uncertaintyPso = PSO(
        objFunc=UncertaintyObjective,
        bounds=bounds,
        popSize=popSize,
        iters=30,
        randomState=None if randomState is None else randomState + 1,
    )
    pointUncertainty, _ = uncertaintyPso.Optimize()

    return pointDiff, pointUncertainty
