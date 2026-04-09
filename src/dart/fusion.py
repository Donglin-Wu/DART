"""
融合模型。
"""

import numpy as np

from .base import Ensure2D
from .kriging import PredictKriging


class FusionModel:
    def __init__(self, krgLow, krgDelta, rho, scalerXHigh, scalerYDelta):
        self.krgLow = krgLow
        self.krgDelta = krgDelta
        self.rho = rho
        self.scalerXHigh = scalerXHigh
        self.scalerYDelta = scalerYDelta

    def Predict(self, xData, returnStd=False, scalerXLow=None, scalerYLow=None):
        xData = Ensure2D(xData)

        xLowScaled = scalerXLow.transform(xData)
        lowMeanScaled = PredictKriging(self.krgLow, xLowScaled, returnStd=False)
        lowMean = scalerYLow.inverse_transform(lowMeanScaled.reshape(-1, 1)).ravel()

        xHighScaled = self.scalerXHigh.transform(xData)
        if returnStd:
            deltaMeanScaled, deltaStdScaled = PredictKriging(self.krgDelta, xHighScaled, returnStd=True)
            deltaVar = deltaStdScaled ** 2 * (self.scalerYDelta.scale_[0] ** 2)
            mixStd = np.sqrt(deltaVar)
        else:
            deltaMeanScaled = PredictKriging(self.krgDelta, xHighScaled, returnStd=False)

        deltaMean = self.scalerYDelta.inverse_transform(deltaMeanScaled.reshape(-1, 1)).ravel()
        mixMean = self.rho * lowMean + deltaMean

        if returnStd:
            return mixMean, mixStd
        return mixMean

    # Compatibility alias for legacy scripts.
    predict = Predict
