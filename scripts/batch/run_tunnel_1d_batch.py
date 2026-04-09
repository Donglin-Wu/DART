from __future__ import annotations

import argparse
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = WORKSPACE_DIR / "src"
INITIAL_DIR = WORKSPACE_DIR / "data" / "raw" / "wind_tunnel" / "initial"
CORRECTED_DIR = WORKSPACE_DIR / "data" / "processed" / "wind_tunnel" / "corrected"
OUTPUT_ROOT = WORKSPACE_DIR / "results" / "run" / "tunnel_1d"
POINT_TOL = 1e-6

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dart.sequential import SequentialFusion  # noqa: E402


BEST_MODEL_CONFIG = {
    "low": {
        "corr": "matern52",
        "poly_type": "constant",
        "nugget_val": 1e-3,
        "l_min": 0.25,
        "l_max": 10.0,
    },
    "delta": {
        "corr": "matern52",
        "poly_type": "linear",
        "nugget_val": 1e-8,
        "l_min": 0.25,
        "l_max": 10.0,
    },
}


@dataclass
class CaseInfo:
    dataName: str
    machName: str
    initialPath: Path
    correctedPath: Path
    sampleCount: int
    fes: int


class AutoTunnelInteractor:
    def __init__(self, correctedPath: Path, tempDir: Path, fes: int):
        self.correctedPath = correctedPath
        self.tempDir = tempDir
        self.fes = int(fes)
        self.correctedFrame = pd.read_csv(correctedPath)
        self.dim = self.correctedFrame.shape[1] - 1
        self.inputColumns = self.correctedFrame.columns[:-1].tolist()
        self.outputColumn = self.correctedFrame.columns[-1]
        self.xTrue = self.correctedFrame.iloc[:, : self.dim].values.astype(float)
        self.yTrue = self.correctedFrame.iloc[:, -1].values.astype(float)
        self.stopRequested = False

    def _FindMatchedRow(self, point):
        pointArray = np.asarray(point, dtype=float).reshape(1, -1)
        distances = np.linalg.norm(self.xTrue - pointArray, axis=1)
        bestIndex = int(np.argmin(distances))
        bestDistance = float(distances[bestIndex])
        if bestDistance > POINT_TOL:
            raise ValueError(f"未在高精度数据中找到匹配坐标，最近距离为 {bestDistance:.3e}，目标点为 {pointArray.ravel().tolist()}")
        return bestIndex

    def _FindMatchedValue(self, point):
        matchIndex = self._FindMatchedRow(point)
        return self.xTrue[matchIndex].copy(), float(self.yTrue[matchIndex])

    def ShowInitialPoints(self, initialPoints):
        print(f"自动处理初始推荐点，共 {len(initialPoints)} 个。")

    def ShowRecommendedPoints(self, pointsToGet):
        pointText = "，".join(str(np.asarray(point, dtype=float).tolist()) for point in pointsToGet)
        print(f"自动处理本轮推荐点：{pointText}")

    def RequestInitialHighCsv(self, initialPoints):
        rows = []
        for point in initialPoints:
            matchedX, matchedY = self._FindMatchedValue(point)
            rows.append(list(matchedX) + [matchedY])

        self.tempDir.mkdir(parents=True, exist_ok=True)
        initialHighPath = self.tempDir / "auto_initial_high.csv"
        pd.DataFrame(rows, columns=self.inputColumns + [self.outputColumn]).to_csv(initialHighPath, index=False)
        print(f"已自动生成初始高精度数据文件：{initialHighPath}")
        return str(initialHighPath)

    def RequestPointValues(self, points, dim, isInitial, xHigh, yHigh):
        responseList = []
        remainingBudget = max(0, self.fes - len(xHigh))
        for pointIndex, point in enumerate(points):
            if pointIndex >= remainingBudget:
                responseList.append("skip")
                continue
            _, matchedY = self._FindMatchedValue(point)
            responseList.append(f"{matchedY:.15g}")
        return responseList, False

    def UpdateState(self, stateData):
        return None

    def NotifyInputError(self, errorMessage):
        print(f"自动输入校验失败：{errorMessage}")

    def ShouldStop(self):
        return self.stopRequested


def IsOneDimensionalCsv(csvPath: Path):
    if "2D" in csvPath.name.upper():
        return False
    dataFrame = pd.read_csv(csvPath, nrows=5)
    return dataFrame.shape[1] == 2


def CollectCases(onlyName=None, onlyMach=None):
    caseList = []
    if not INITIAL_DIR.exists() or not CORRECTED_DIR.exists():
        raise FileNotFoundError("未找到 WindTunnelData 下的 InitialData 或 CorrectedData 目录。")

    for initialDataDir in sorted(path for path in INITIAL_DIR.iterdir() if path.is_dir()):
        dataName = initialDataDir.name
        if onlyName and dataName != onlyName:
            continue

        correctedDataDir = CORRECTED_DIR / dataName
        if not correctedDataDir.exists():
            print(f"跳过 {dataName}：未找到对应的高精度目录。")
            continue

        for initialCsv in sorted(initialDataDir.glob("Ma*.csv")):
            machName = initialCsv.stem
            if onlyMach and machName != onlyMach:
                continue
            if not IsOneDimensionalCsv(initialCsv):
                continue

            correctedCsv = correctedDataDir / initialCsv.name
            if not correctedCsv.exists():
                print(f"跳过 {dataName}/{machName}：未找到对应的高精度文件。")
                continue
            if not IsOneDimensionalCsv(correctedCsv):
                continue

            initialFrame = pd.read_csv(initialCsv)
            correctedFrame = pd.read_csv(correctedCsv)
            sampleCount = min(len(initialFrame), len(correctedFrame))
            # fes = max(6, int(math.ceil(sampleCount * 0.5)))
            fes = max(6, int(math.ceil(sampleCount * 0.48)))

            caseList.append(
                CaseInfo(
                    dataName=dataName,
                    machName=machName,
                    initialPath=initialCsv,
                    correctedPath=correctedCsv,
                    sampleCount=sampleCount,
                    fes=fes,
                )
            )

    return caseList


def ComputeMetrics(predictionCsvPath: Path, truthCsvPath: Path):
    predictionFrame = pd.read_csv(predictionCsvPath)
    truthFrame = pd.read_csv(truthCsvPath)
    inputColumns = truthFrame.columns[:-1].tolist()
    truthColumn = truthFrame.columns[-1]

    mergedFrame = predictionFrame[inputColumns + ["fusion_prediction"]].merge(
        truthFrame[inputColumns + [truthColumn]],
        on=inputColumns,
        how="inner",
    )
    if mergedFrame.empty:
        raise ValueError("预测结果与高精度验证数据无法按输入坐标匹配。")

    yPred = mergedFrame["fusion_prediction"].to_numpy(dtype=float)
    yTrue = mergedFrame[truthColumn].to_numpy(dtype=float)
    rmseValue = float(np.sqrt(np.mean((yPred - yTrue) ** 2)))
    maeValue = float(np.mean(np.abs(yPred - yTrue)))
    residualSum = float(np.sum((yTrue - yPred) ** 2))
    totalSum = float(np.sum((yTrue - np.mean(yTrue)) ** 2))
    r2Value = 1.0 - residualSum / totalSum if totalSum > 1e-12 else 1.0
    return rmseValue, maeValue, r2Value, len(mergedFrame)


def RunCase(caseInfo: CaseInfo):
    print("=" * 80)
    print(f"开始处理：{caseInfo.dataName} / {caseInfo.machName}")
    print(f"低精度文件：{caseInfo.initialPath}")
    print(f"高精度文件：{caseInfo.correctedPath}")
    print(f"样本数：{caseInfo.sampleCount}，Fes：{caseInfo.fes}")

    caseOutputBase = OUTPUT_ROOT / caseInfo.dataName / caseInfo.machName
    interactor = AutoTunnelInteractor(caseInfo.correctedPath, caseOutputBase, caseInfo.fes)
    resultInfo = SequentialFusion(
        cfdCsv=str(caseInfo.initialPath),
        fes=caseInfo.fes,
        maxIters=max(caseInfo.fes, 50),
        randomState=42,
        krgCorr="matern52",
        validationCsv=str(caseInfo.correctedPath),
        modelConfig=BEST_MODEL_CONFIG,
        outputBaseDir=str(caseOutputBase),
        interactor=interactor,
    )

    predictionCsvPath = Path(resultInfo["predictionCsvPath"])
    rmseValue, maeValue, r2Value, matchedCount = ComputeMetrics(predictionCsvPath, caseInfo.correctedPath)
    print(f"处理完成：RMSE={rmseValue:.6f}，MAE={maeValue:.6f}，R2={r2Value:.6f}，匹配样本数={matchedCount}")

    return {
        "data_name": caseInfo.dataName,
        "mach_name": caseInfo.machName,
        "initial_csv": str(caseInfo.initialPath),
        "corrected_csv": str(caseInfo.correctedPath),
        "sample_count": caseInfo.sampleCount,
        "fes": caseInfo.fes,
        "result_dir": str(resultInfo["resultDir"]),
        "prediction_csv": str(predictionCsvPath),
        "scatter_svg": str(resultInfo["scatterSvgPath"]) if resultInfo["scatterSvgPath"] else None,
        "visualization_svg": str(resultInfo["visualizationPath"]) if resultInfo["visualizationPath"] else None,
        "model_pkl": str(resultInfo["modelPath"]) if resultInfo["modelPath"] else None,
        "rmse": rmseValue,
        "mae": maeValue,
        "r2": r2Value,
        "matched_count": matchedCount,
        "status": "success",
    }


def Main():
    parser = argparse.ArgumentParser(description="批量处理风洞一维双精度融合数据。")
    parser.add_argument("--only-name", help="仅处理指定数据名称文件夹。")
    parser.add_argument("--only-mach", help="仅处理指定马赫数文件，例如 Ma0.3。")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    caseList = CollectCases(onlyName=args.only_name, onlyMach=args.only_mach)

    if not caseList:
        print("未找到可处理的一维数据。")
        return

    print(f"共找到 {len(caseList)} 个一维融合工况。")

    summaryRows = []
    failedRows = []

    for caseInfo in caseList:
        try:
            summaryRows.append(RunCase(caseInfo))
        except Exception as exc:
            failedInfo = {
                "data_name": caseInfo.dataName,
                "mach_name": caseInfo.machName,
                "initial_csv": str(caseInfo.initialPath),
                "corrected_csv": str(caseInfo.correctedPath),
                "sample_count": caseInfo.sampleCount,
                "fes": caseInfo.fes,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failedRows.append(failedInfo)
            print(f"处理失败：{caseInfo.dataName} / {caseInfo.machName}，错误：{exc}")

    summaryPath = OUTPUT_ROOT / "tunnel1d_summary.csv"
    summaryColumns = [
        "data_name",
        "mach_name",
        "initial_csv",
        "corrected_csv",
        "sample_count",
        "fes",
        "result_dir",
        "prediction_csv",
        "scatter_svg",
        "visualization_svg",
        "model_pkl",
        "rmse",
        "mae",
        "r2",
        "matched_count",
        "status",
    ]
    pd.DataFrame(summaryRows, columns=summaryColumns).to_csv(summaryPath, index=False, encoding="utf-8-sig")

    failedPath = OUTPUT_ROOT / "tunnel1d_failures.csv"
    failedColumns = [
        "data_name",
        "mach_name",
        "initial_csv",
        "corrected_csv",
        "sample_count",
        "fes",
        "status",
        "error",
        "traceback",
    ]
    pd.DataFrame(failedRows, columns=failedColumns).to_csv(failedPath, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"批处理结束。成功 {len(summaryRows)} 个，失败 {len(failedRows)} 个。")
    print(f"汇总文件：{summaryPath}")
    print(f"失败文件：{failedPath}")


if __name__ == "__main__":
    Main()
