"""
Basic utilities.
"""

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd


def SetupLogger():
    """
    Initialize logging.
    """
    logger = logging.getLogger("FusionModel")
    logger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(logging.Formatter("%(message)s"))

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(consoleHandler)
    return logger


logger = SetupLogger()


def ReadCsvAuto(path):
    """
    Read a CSV file; the last column is treated as the output by default.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

    dataFrame = pd.read_csv(path)
    if dataFrame.shape[1] < 2:
        raise ValueError("CSV must contain at least one input column and one output column.")

    xData = dataFrame.iloc[:, :-1].values.astype(float)
    yData = dataFrame.iloc[:, -1].values.astype(float)
    inputColumns = dataFrame.columns[:-1].tolist()
    outputColumn = dataFrame.columns[-1]
    return xData, yData, inputColumns, outputColumn


def Ensure2D(xData):
    """
    Ensure the input is a two-dimensional array.
    """
    arrayData = np.asarray(xData)
    if arrayData.ndim == 1:
        return arrayData.reshape(1, -1)
    return arrayData


def CreateResultDirectory(baseDir=None):
    """
    Create the result directory for this run.
    """
    rootDir = baseDir or os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resultDir = os.path.join(rootDir, f"FusionResult{timestamp}")
    os.makedirs(resultDir, exist_ok=True)
    return resultDir, timestamp
