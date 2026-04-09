"""
基础工具。
"""

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd


def SetupLogger():
    """
    初始化日志。
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
    读取 CSV，默认最后一列为输出列。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    dataFrame = pd.read_csv(path)
    if dataFrame.shape[1] < 2:
        raise ValueError("CSV 至少需要 1 列输入和 1 列输出。")

    xData = dataFrame.iloc[:, :-1].values.astype(float)
    yData = dataFrame.iloc[:, -1].values.astype(float)
    inputColumns = dataFrame.columns[:-1].tolist()
    outputColumn = dataFrame.columns[-1]
    return xData, yData, inputColumns, outputColumn


def Ensure2D(xData):
    """
    确保输入是二维数组。
    """
    arrayData = np.asarray(xData)
    if arrayData.ndim == 1:
        return arrayData.reshape(1, -1)
    return arrayData


def CreateResultDirectory(baseDir=None):
    """
    创建本次运行的结果目录。
    """
    rootDir = baseDir or os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resultDir = os.path.join(rootDir, f"FusionResult{timestamp}")
    os.makedirs(resultDir, exist_ok=True)
    return resultDir, timestamp


# Compatibility aliases for legacy scripts.
read_csv_auto = ReadCsvAuto
