"""
基于训练好的 pkl 模型生成预测结果和回归拟合散点图。
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .base import CreateResultDirectory


def PredictRows(fusionModel, xData, scalerXLow, scalerYLow):
    predictions = []
    for rowData in xData:
        predValue = fusionModel.Predict(
            np.asarray(rowData, dtype=float).reshape(1, -1),
            scalerXLow=scalerXLow,
            scalerYLow=scalerYLow,
        )
        predictions.append(float(np.asarray(predValue).ravel()[0]))
    return np.asarray(predictions, dtype=float)


def ComputeMetrics(yTrue, yPred):
    yTrue = np.asarray(yTrue, dtype=float)
    yPred = np.asarray(yPred, dtype=float)
    rmseValue = float(np.sqrt(np.mean((yTrue - yPred) ** 2)))
    maeValue = float(np.mean(np.abs(yTrue - yPred)))
    residualSum = float(np.sum((yTrue - yPred) ** 2))
    totalSum = float(np.sum((yTrue - np.mean(yTrue)) ** 2))
    r2Value = 1.0 - residualSum / totalSum if totalSum > 1e-12 else 1.0
    return rmseValue, maeValue, r2Value


def SaveScatterSvg(outputPath, yTrue, yPred, rmseValue, maeValue, r2Value, xLabel, yLabel):
    yTrue = np.asarray(yTrue, dtype=float)
    yPred = np.asarray(yPred, dtype=float)

    allValues = np.concatenate([yTrue, yPred])
    minValue = float(np.min(allValues))
    maxValue = float(np.max(allValues))
    if abs(maxValue - minValue) < 1e-12:
        maxValue = minValue + 1.0

    padding = 0.05 * (maxValue - minValue)
    xMin = minValue - padding
    xMax = maxValue + padding
    yMin = xMin
    yMax = xMax

    width = 1000
    height = 800
    left = 100
    right = 40
    top = 70
    bottom = 90
    plotWidth = width - left - right
    plotHeight = height - top - bottom

    def ScaleX(xValue):
        return left + (float(xValue) - xMin) / (xMax - xMin) * plotWidth

    def ScaleY(yValue):
        return top + (yMax - float(yValue)) / (yMax - yMin) * plotHeight

    pointsSvg = "\n".join(
        f'<circle cx="{ScaleX(xValue)}" cy="{ScaleY(yValue)}" r="5" fill="#1f77b4" opacity="0.75" />'
        for xValue, yValue in zip(yTrue, yPred)
    )

    diagonalLine = (
        f'<line x1="{ScaleX(xMin)}" y1="{ScaleY(yMin)}" x2="{ScaleX(xMax)}" y2="{ScaleY(yMax)}" '
        f'stroke="#d62728" stroke-width="2" stroke-dasharray="8,6" />'
    )
    metricsText = f"RMSE = {rmseValue:.6f}    MAE = {maeValue:.6f}    R2 = {r2Value:.6f}"

    tickLines = []
    for tickValue in np.linspace(xMin, xMax, 6):
        tickLines.append(
            f'<line x1="{ScaleX(tickValue)}" y1="{height - bottom}" x2="{ScaleX(tickValue)}" y2="{height - bottom + 8}" stroke="black" />'
            f'<text x="{ScaleX(tickValue)}" y="{height - bottom + 30}" text-anchor="middle" font-family="Arial" font-size="14">{tickValue:.3f}</text>'
        )
        tickLines.append(
            f'<line x1="{left - 8}" y1="{ScaleY(tickValue)}" x2="{left}" y2="{ScaleY(tickValue)}" stroke="black" />'
            f'<text x="{left - 15}" y="{ScaleY(tickValue) + 5}" text-anchor="end" font-family="Arial" font-size="14">{tickValue:.3f}</text>'
        )

    svgContent = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width / 2}" y="35" text-anchor="middle" font-family="Arial" font-size="24">回归拟合散点图</text>
<text x="{width / 2}" y="62" text-anchor="middle" font-family="Arial" font-size="16">{metricsText}</text>
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="black" stroke-width="2" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="black" stroke-width="2" />
{''.join(tickLines)}
{diagonalLine}
{pointsSvg}
<text x="{width / 2}" y="{height - 25}" text-anchor="middle" font-family="Arial" font-size="18">{xLabel}</text>
<text x="30" y="{height / 2}" transform="rotate(-90 30,{height / 2})" text-anchor="middle" font-family="Arial" font-size="18">{yLabel}</text>
</svg>"""
    outputPath.write_text(svgContent, encoding="utf-8")


def BuildScatterData(fusionModel, scalerXLow, scalerYLow, truthDataFrame, dim):
    xTrue = truthDataFrame.iloc[:, :dim].values.astype(float)
    yTrue = truthDataFrame.iloc[:, -1].values.astype(float)
    yPred = PredictRows(fusionModel, xTrue, scalerXLow, scalerYLow)
    return yTrue, yPred


def GeneratePredictionAndScatter(modelPath, inputCsvPath, outputCsvPath, truthCsvPath=None, scatterSvgPath=None):
    modelPath = Path(modelPath)
    inputCsvPath = Path(inputCsvPath)
    outputCsvPath = Path(outputCsvPath)
    scatterSvgPath = Path(scatterSvgPath) if scatterSvgPath else outputCsvPath.with_name("regression_scatter.svg")
    truthCsvPath = Path(truthCsvPath) if truthCsvPath else None

    with modelPath.open("rb") as modelFile:
        modelData = pickle.load(modelFile)

    fusionModel = modelData["fusion_model"]
    scalerXLow = modelData["scaler_X_low"]
    scalerYLow = modelData["scaler_y_low"]
    dim = int(modelData["dim"])

    inputDataFrame = pd.read_csv(inputCsvPath)
    xInput = inputDataFrame.iloc[:, :dim].values.astype(float)
    yPredInput = PredictRows(fusionModel, xInput, scalerXLow, scalerYLow)

    outputDataFrame = inputDataFrame.copy()
    outputDataFrame["fusion_prediction"] = yPredInput
    outputDataFrame.to_csv(outputCsvPath, index=False)

    scatterSourceFrame = pd.read_csv(truthCsvPath) if truthCsvPath else inputDataFrame
    yTrue, yPredScatter = BuildScatterData(
        fusionModel=fusionModel,
        scalerXLow=scalerXLow,
        scalerYLow=scalerYLow,
        truthDataFrame=scatterSourceFrame,
        dim=dim,
    )
    rmseValue, maeValue, r2Value = ComputeMetrics(yTrue, yPredScatter)
    SaveScatterSvg(
        outputPath=scatterSvgPath,
        yTrue=yTrue,
        yPred=yPredScatter,
        rmseValue=rmseValue,
        maeValue=maeValue,
        r2Value=r2Value,
        xLabel="真实的物理量",
        yLabel="融合模型的预测量",
    )

    return {
        "predictionCsv": str(outputCsvPath),
        "scatterSvg": str(scatterSvgPath),
        "rmse": rmseValue,
        "mae": maeValue,
        "r2": r2Value,
    }


def Main():
    parser = argparse.ArgumentParser(description="基于训练好的 pkl 模型输出预测结果和回归拟合散点图。")
    parser.add_argument("--model", required=True, help="训练得到的 trained_model.pkl 路径")
    parser.add_argument("--input", required=True, help="用于生成预测结果的输入 CSV 路径")
    parser.add_argument("--truth", help="用于回归拟合散点图的真实高精度 CSV 路径")
    parser.add_argument("--output-dir", help="输出目录，不传时自动新建 FusionResult 时间戳目录")
    parser.add_argument("--output-csv", help="预测结果 CSV 路径")
    parser.add_argument("--scatter-svg", help="回归拟合散点图 SVG 路径")
    args = parser.parse_args()

    if args.output_dir:
        resultDir = Path(args.output_dir)
        resultDir.mkdir(parents=True, exist_ok=True)
    else:
        resultDirString, _ = CreateResultDirectory()
        resultDir = Path(resultDirString)

    outputCsvPath = Path(args.output_csv) if args.output_csv else resultDir / "fusion_predictions.csv"
    scatterSvgPath = Path(args.scatter_svg) if args.scatter_svg else resultDir / "regression_scatter.svg"

    resultInfo = GeneratePredictionAndScatter(
        modelPath=args.model,
        inputCsvPath=args.input,
        outputCsvPath=outputCsvPath,
        truthCsvPath=args.truth,
        scatterSvgPath=scatterSvgPath,
    )

    print(f"预测结果已保存: {resultInfo['predictionCsv']}")
    print(f"回归拟合散点图已保存: {resultInfo['scatterSvg']}")
    print(f"RMSE={resultInfo['rmse']:.6f}, MAE={resultInfo['mae']:.6f}, R2={resultInfo['r2']:.6f}")


if __name__ == "__main__":
    Main()
