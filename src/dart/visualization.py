"""
融合结果可视化。
"""

import base64
import io
import os

import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D


matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def Save1DVisualizationSvg(outputPath, xPlot, meanFusion, stdFusion, xLow, yLow, xHigh, yHigh, fes, iterations, xVal=None, yVal=None, xLabel="输入变量", yLabel="输出变量"):
    width = 1200
    height = 800
    left = 110
    right = 40
    top = 70
    bottom = 90
    plotWidth = width - left - right
    plotHeight = height - top - bottom

    allX = [float(value) for value in np.asarray(xPlot).ravel()]
    allY = [float(value) for value in np.asarray(meanFusion).ravel()]
    allY.extend(float(value) for value in np.asarray(yLow).ravel())
    allY.extend(float(value) for value in np.asarray(yHigh).ravel())
    if xVal is not None and yVal is not None:
        allX.extend(float(value) for value in np.asarray(xVal).ravel())
        allY.extend(float(value) for value in np.asarray(yVal).ravel())

    xMin = min(allX)
    xMax = max(allX)
    yMin = min(allY)
    yMax = max(allY)

    if abs(xMax - xMin) < 1e-12:
        xMax = xMin + 1.0
    if abs(yMax - yMin) < 1e-12:
        yMax = yMin + 1.0

    yMin = min(yMin, float(np.min(meanFusion - 1.96 * stdFusion)))
    yMax = max(yMax, float(np.max(meanFusion + 1.96 * stdFusion)))
    yPadding = 0.06 * (yMax - yMin)
    yMin -= yPadding
    yMax += yPadding

    def ScaleX(xValue):
        return left + (float(xValue) - xMin) / (xMax - xMin) * plotWidth

    def ScaleY(yValue):
        return top + (yMax - float(yValue)) / (yMax - yMin) * plotHeight

    meanPath = " ".join(
        ("M" if index == 0 else "L") + f" {ScaleX(xValue):.2f} {ScaleY(yValue):.2f}"
        for index, (xValue, yValue) in enumerate(zip(np.asarray(xPlot).ravel(), np.asarray(meanFusion).ravel()))
    )
    upperPoints = [(ScaleX(xValue), ScaleY(yValue)) for xValue, yValue in zip(np.asarray(xPlot).ravel(), np.asarray(meanFusion + 1.96 * stdFusion).ravel())]
    lowerPoints = [(ScaleX(xValue), ScaleY(yValue)) for xValue, yValue in zip(np.asarray(xPlot).ravel()[::-1], np.asarray(meanFusion - 1.96 * stdFusion).ravel()[::-1])]
    bandPoints = upperPoints + lowerPoints
    bandPolygon = " ".join(f"{xCoord:.2f},{yCoord:.2f}" for xCoord, yCoord in bandPoints)

    lowPointsSvg = "\n".join(
        f'<circle cx="{ScaleX(xValue):.2f}" cy="{ScaleY(yValue):.2f}" r="4" fill="#4f81bd" opacity="0.55" />'
        for xValue, yValue in zip(xLow[:, 0], yLow)
    )
    highPointsSvg = "\n".join(
        f'<circle cx="{ScaleX(xValue):.2f}" cy="{ScaleY(yValue):.2f}" r="5" fill="#c0504d" />'
        for xValue, yValue in zip(xHigh[:, 0], yHigh)
    )

    truthPath = ""
    if xVal is not None and yVal is not None:
        sortIndex = np.argsort(np.asarray(xVal).ravel())
        truthPath = " ".join(
            ("M" if index == 0 else "L") + f" {ScaleX(xValue):.2f} {ScaleY(yValue):.2f}"
            for index, (xValue, yValue) in enumerate(zip(np.asarray(xVal).ravel()[sortIndex], np.asarray(yVal).ravel()[sortIndex]))
        )

    tickSvg = []
    for xTick in np.linspace(xMin, xMax, 6):
        tickSvg.append(
            f'<line x1="{ScaleX(xTick):.2f}" y1="{height - bottom}" x2="{ScaleX(xTick):.2f}" y2="{height - bottom + 8}" stroke="black" />'
            f'<text x="{ScaleX(xTick):.2f}" y="{height - bottom + 30}" text-anchor="middle" font-family="Arial" font-size="14">{xTick:.3f}</text>'
        )
    for yTick in np.linspace(yMin, yMax, 6):
        tickSvg.append(
            f'<line x1="{left - 8}" y1="{ScaleY(yTick):.2f}" x2="{left}" y2="{ScaleY(yTick):.2f}" stroke="black" />'
            f'<text x="{left - 15}" y="{ScaleY(yTick) + 5:.2f}" text-anchor="end" font-family="Arial" font-size="14">{yTick:.3f}</text>'
        )

    truthSvg = f'<path d="{truthPath}" fill="none" stroke="black" stroke-width="2" opacity="0.85" />' if truthPath else ""

    svgContent = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width / 2}" y="35" text-anchor="middle" font-family="Arial" font-size="24">双精度融合结果图</text>
<text x="{width / 2}" y="62" text-anchor="middle" font-family="Arial" font-size="16">Fes = {fes}    迭代次数 = {iterations}</text>
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="black" stroke-width="2" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="black" stroke-width="2" />
{''.join(tickSvg)}
<polygon points="{bandPolygon}" fill="#2f7ed8" opacity="0.16" />
{truthSvg}
<path d="{meanPath}" fill="none" stroke="#2f7ed8" stroke-width="2.5" />
{lowPointsSvg}
{highPointsSvg}
<text x="{width / 2}" y="{height - 25}" text-anchor="middle" font-family="Arial" font-size="18">{xLabel}</text>
<text x="30" y="{height / 2}" transform="rotate(-90 30,{height / 2})" text-anchor="middle" font-family="Arial" font-size="18">{yLabel}</text>
<rect x="{width - 280}" y="90" width="18" height="18" fill="#2f7ed8" opacity="0.16" />
<text x="{width - 252}" y="104" font-family="Arial" font-size="14">95% 置信区间</text>
<line x1="{width - 280}" y1="132" x2="{width - 258}" y2="132" stroke="#2f7ed8" stroke-width="2.5" />
<text x="{width - 252}" y="136" font-family="Arial" font-size="14">融合模型</text>
<circle cx="{width - 271}" cy="160" r="4" fill="#4f81bd" opacity="0.55" />
<text x="{width - 252}" y="165" font-family="Arial" font-size="14">低精度样本</text>
<circle cx="{width - 271}" cy="188" r="5" fill="#c0504d" />
<text x="{width - 252}" y="193" font-family="Arial" font-size="14">高精度样本</text>
<line x1="{width - 280}" y1="216" x2="{width - 258}" y2="216" stroke="black" stroke-width="2" />
<text x="{width - 252}" y="220" font-family="Arial" font-size="14">真实高精度曲线</text>
</svg>"""
    with open(outputPath, "w", encoding="utf-8") as svgFile:
        svgFile.write(svgContent)


def PredictFusionRows(fusionModel, xData, scalerXLow, scalerYLow, returnStd=False):
    meanList = []
    stdList = []

    for rowData in xData:
        predValue = fusionModel.Predict(
            np.asarray(rowData, dtype=float).reshape(1, -1),
            returnStd=returnStd,
            scalerXLow=scalerXLow,
            scalerYLow=scalerYLow,
        )
        if returnStd:
            meanValue, stdValue = predValue
            meanList.append(float(np.asarray(meanValue).ravel()[0]))
            stdList.append(float(np.asarray(stdValue).ravel()[0]))
        else:
            meanList.append(float(np.asarray(predValue).ravel()[0]))

    if returnStd:
        return np.asarray(meanList, dtype=float), np.asarray(stdList, dtype=float)
    return np.asarray(meanList, dtype=float)


def Project3DPoints(points3d, azimuthDeg=-58.0, elevationDeg=28.0):
    points3d = np.asarray(points3d, dtype=float)
    if len(points3d) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    centered = points3d - np.mean(points3d, axis=0, keepdims=True)
    scaleVector = np.ptp(points3d, axis=0)
    scaleVector[scaleVector < 1e-12] = 1.0
    normalized = centered / scaleVector
    normalized[:, 2] *= 1.4

    azimuth = np.deg2rad(azimuthDeg)
    elevation = np.deg2rad(elevationDeg)

    cosAzimuth = np.cos(azimuth)
    sinAzimuth = np.sin(azimuth)
    cosElevation = np.cos(elevation)
    sinElevation = np.sin(elevation)

    xValue = normalized[:, 0]
    yValue = normalized[:, 1]
    zValue = normalized[:, 2]

    xRotate = cosAzimuth * xValue - sinAzimuth * yValue
    yRotate = sinAzimuth * xValue + cosAzimuth * yValue
    zRotate = zValue

    xFinal = xRotate
    yFinal = cosElevation * yRotate - sinElevation * zRotate
    zFinal = sinElevation * yRotate + cosElevation * zRotate

    projected = np.empty((len(points3d), 2), dtype=float)
    projected[:, 0] = xFinal
    projected[:, 1] = zFinal
    depthValues = yFinal
    return projected, depthValues


def Save2DVisualizationSvg(outputPath, xLow, yLow, xHigh, yHigh, xGrid, yGrid, zGrid, fes, iterations, xVal=None, yVal=None, inputLabels=None, outputLabel="输出变量"):
    width = 1400
    height = 1000
    left = 90
    right = 260
    top = 70
    bottom = 70
    plotWidth = width - left - right
    plotHeight = height - top - bottom

    gridPoints3d = np.column_stack([xGrid.ravel(), yGrid.ravel(), zGrid.ravel()])
    projectedGrid, depthGrid = Project3DPoints(gridPoints3d)
    projX = projectedGrid[:, 0]
    projY = projectedGrid[:, 1]

    minProjX = float(np.min(projX))
    maxProjX = float(np.max(projX))
    minProjY = float(np.min(projY))
    maxProjY = float(np.max(projY))

    def ScaleX(xValue):
        return left + (float(xValue) - minProjX) / (maxProjX - minProjX + 1e-12) * plotWidth

    def ScaleY(yValue):
        return top + (maxProjY - float(yValue)) / (maxProjY - minProjY + 1e-12) * plotHeight

    polygonItems = []
    gridRows, gridCols = zGrid.shape
    colorMap = cm.get_cmap("viridis")
    zMin = float(np.min(zGrid))
    zMax = float(np.max(zGrid))
    zSpan = max(zMax - zMin, 1e-12)

    for rowIndex in range(gridRows - 1):
        for colIndex in range(gridCols - 1):
            index00 = rowIndex * gridCols + colIndex
            index01 = rowIndex * gridCols + colIndex + 1
            index10 = (rowIndex + 1) * gridCols + colIndex
            index11 = (rowIndex + 1) * gridCols + colIndex + 1

            cornerIndexes = [index00, index01, index11, index10]
            polygonPoints = " ".join(
                f"{ScaleX(projectedGrid[indexValue, 0]):.2f},{ScaleY(projectedGrid[indexValue, 1]):.2f}"
                for indexValue in cornerIndexes
            )
            averageDepth = float(np.mean(depthGrid[cornerIndexes]))
            averageZ = float(np.mean(gridPoints3d[cornerIndexes, 2]))
            rgbaValue = colorMap((averageZ - zMin) / zSpan)
            fillColor = "#{:02x}{:02x}{:02x}".format(
                int(rgbaValue[0] * 255),
                int(rgbaValue[1] * 255),
                int(rgbaValue[2] * 255),
            )
            polygonItems.append((averageDepth, polygonPoints, fillColor))

    polygonItems.sort(key=lambda item: item[0])
    polygonSvg = "\n".join(
        f'<polygon points="{polygonPoints}" fill="{fillColor}" stroke="{fillColor}" stroke-width="0.35" opacity="0.92" />'
        for _, polygonPoints, fillColor in polygonItems
    )

    def BuildProjectedCircles(xData, yData, colorValue, radiusValue, opacityValue, borderColor="none"):
        if xData is None or yData is None or len(xData) == 0:
            return ""
        points3d = np.column_stack([xData[:, 0], xData[:, 1], yData])
        projectedPoints, depthValues = Project3DPoints(points3d)
        drawOrder = np.argsort(depthValues)
        return "\n".join(
            f'<circle cx="{ScaleX(projectedPoints[indexValue, 0]):.2f}" cy="{ScaleY(projectedPoints[indexValue, 1]):.2f}" r="{radiusValue}" fill="{colorValue}" opacity="{opacityValue}" stroke="{borderColor}" stroke-width="0.8" />'
            for indexValue in drawOrder
        )

    lowSvg = BuildProjectedCircles(xLow, yLow, "#4f81bd", 2.7, 0.38)
    validationSvg = BuildProjectedCircles(xVal, yVal, "#808080", 2.2, 0.26) if xVal is not None and yVal is not None else ""
    highSvg = BuildProjectedCircles(xHigh, yHigh, "#c0504d", 4.0, 0.96, borderColor="#000000")

    xLabel = inputLabels[0] if inputLabels and len(inputLabels) > 0 else "输入变量1"
    yLabel = inputLabels[1] if inputLabels and len(inputLabels) > 1 else "输入变量2"
    zLabel = outputLabel or "输出变量"

    svgContent = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width / 2}" y="36" text-anchor="middle" font-family="Microsoft YaHei UI, Arial" font-size="26">双精度融合三维结果图</text>
<text x="{width / 2}" y="66" text-anchor="middle" font-family="Microsoft YaHei UI, Arial" font-size="16">Fes = {fes}    迭代次数 = {iterations}</text>
<text x="{left}" y="{height - 26}" font-family="Microsoft YaHei UI, Arial" font-size="18">{xLabel}</text>
<text x="{left}" y="{height - 52}" font-family="Microsoft YaHei UI, Arial" font-size="18">{yLabel}</text>
<text x="{left}" y="{height - 78}" font-family="Microsoft YaHei UI, Arial" font-size="18">{zLabel}</text>
{polygonSvg}
{lowSvg}
{validationSvg}
{highSvg}
<rect x="{width - 220}" y="110" width="20" height="20" fill="#2f7ed8" opacity="0.95" />
<text x="{width - 190}" y="125" font-family="Microsoft YaHei UI, Arial" font-size="15">融合曲面</text>
<circle cx="{width - 210}" cy="157" r="4" fill="#4f81bd" opacity="0.5" />
<text x="{width - 190}" y="162" font-family="Microsoft YaHei UI, Arial" font-size="15">低精度样本</text>
<circle cx="{width - 210}" cy="189" r="4" fill="#c0504d" stroke="#000000" stroke-width="0.8" />
<text x="{width - 190}" y="194" font-family="Microsoft YaHei UI, Arial" font-size="15">高精度样本</text>
<circle cx="{width - 210}" cy="221" r="4" fill="#808080" opacity="0.35" />
<text x="{width - 190}" y="226" font-family="Microsoft YaHei UI, Arial" font-size="15">验证数据</text>
</svg>"""

    with open(outputPath, "w", encoding="utf-8") as svgFile:
        svgFile.write(svgContent)


def VisualizeResults(xLow, yLow, xHigh, yHigh, fusionModel, scalerXLow, scalerYLow, bounds, dim, iterations, fes, outputDir, xVal=None, yVal=None, inputLabels=None, outputLabel="输出变量"):
    """
    输出 1 维或 2 维融合结果图。
    """
    outputPath = os.path.join(outputDir, "fusion_visualization.svg")

    if dim == 1:
        xMin, xMax = bounds[0]
        xPlot = np.linspace(xMin, xMax, 300).reshape(-1, 1)
        meanFusion, stdFusion = PredictFusionRows(
            fusionModel=fusionModel,
            xData=xPlot,
            scalerXLow=scalerXLow,
            scalerYLow=scalerYLow,
            returnStd=True,
        )

        xLabel = inputLabels[0] if inputLabels else "输入变量"
        yLabel = outputLabel or "输出变量"
        Save1DVisualizationSvg(
            outputPath=outputPath,
            xPlot=xPlot,
            meanFusion=meanFusion,
            stdFusion=stdFusion,
            xLow=xLow,
            yLow=yLow,
            xHigh=xHigh,
            yHigh=yHigh,
            fes=fes,
            iterations=iterations,
            xVal=xVal,
            yVal=yVal,
            xLabel=xLabel,
            yLabel=yLabel,
        )
        return outputPath

    if dim == 2:
        x1Min, x1Max = bounds[0]
        x2Min, x2Max = bounds[1]
        x1Plot = np.linspace(x1Min, x1Max, 45)
        x2Plot = np.linspace(x2Min, x2Max, 45)
        meshX1, meshX2 = np.meshgrid(x1Plot, x2Plot)
        gridPoints = np.vstack([meshX1.ravel(), meshX2.ravel()]).T

        meanFusion, _ = PredictFusionRows(
            fusionModel=fusionModel,
            xData=gridPoints,
            scalerXLow=scalerXLow,
            scalerYLow=scalerYLow,
            returnStd=True,
        )
        meshZ = meanFusion.reshape(meshX1.shape)
        Save2DVisualizationSvg(
            outputPath=outputPath,
            xLow=xLow,
            yLow=yLow,
            xHigh=xHigh,
            yHigh=yHigh,
            xGrid=meshX1,
            yGrid=meshX2,
            zGrid=meshZ,
            fes=fes,
            iterations=iterations,
            xVal=xVal,
            yVal=yVal,
            inputLabels=inputLabels,
            outputLabel=outputLabel,
        )
        return outputPath

    return None
