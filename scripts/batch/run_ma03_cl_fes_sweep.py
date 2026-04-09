from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = WORKSPACE_DIR / "src"
OUTPUT_ROOT = WORKSPACE_DIR / "results" / "run" / "ma03_cl"
INITIAL_CSV = WORKSPACE_DIR / "data" / "raw" / "wind_tunnel" / "initial" / "CL" / "Ma0.3.csv"
CORRECTED_CSV = WORKSPACE_DIR / "data" / "processed" / "wind_tunnel" / "corrected" / "CL" / "Ma0.3.csv"
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
class SweepCase:
    fes: int
    caseDir: Path


class AutoHighFidelityInteractor:
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
        print(f"自动处理初始高精度点，共 {len(initialPoints)} 个。")

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
        print(f"已自动生成初始高精度文件：{initialHighPath}")
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
        return False


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


def NormalizeSeries(series: pd.Series):
    minValue = float(series.min())
    maxValue = float(series.max())
    if maxValue - minValue <= 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - minValue) / (maxValue - minValue)


def AddCompositeScores(summaryFrame: pd.DataFrame):
    scoredFrame = summaryFrame.copy()
    scoredFrame["rmse_norm"] = NormalizeSeries(scoredFrame["rmse"])
    scoredFrame["mae_norm"] = NormalizeSeries(scoredFrame["mae"])
    scoredFrame["fes_norm"] = NormalizeSeries(scoredFrame["fes"])
    scoredFrame["r2_penalty"] = 1.0 - scoredFrame["r2"]
    scoredFrame["r2_penalty_norm"] = NormalizeSeries(scoredFrame["r2_penalty"])

    scoredFrame["score_error_first"] = (
        0.60 * scoredFrame["rmse_norm"]
        + 0.15 * scoredFrame["mae_norm"]
        + 0.15 * scoredFrame["r2_penalty_norm"]
        + 0.10 * scoredFrame["fes_norm"]
    )
    scoredFrame["score_budget_balanced"] = (
        0.45 * scoredFrame["rmse_norm"]
        + 0.15 * scoredFrame["mae_norm"]
        + 0.10 * scoredFrame["r2_penalty_norm"]
        + 0.30 * scoredFrame["fes_norm"]
    )
    return scoredFrame


def RunOneCase(case: SweepCase):
    print("=" * 80)
    print(f"开始处理 CL / Ma0.3，Fes={case.fes}")
    print(f"输出目录基路径：{case.caseDir}")

    interactor = AutoHighFidelityInteractor(CORRECTED_CSV, case.caseDir, case.fes)
    resultInfo = SequentialFusion(
        cfdCsv=str(INITIAL_CSV),
        fes=case.fes,
        maxIters=max(case.fes, 50),
        randomState=42,
        krgCorr="matern52",
        validationCsv=str(CORRECTED_CSV),
        modelConfig=BEST_MODEL_CONFIG,
        outputBaseDir=str(case.caseDir),
        interactor=interactor,
    )

    predictionCsvPath = Path(resultInfo["predictionCsvPath"])
    rmseValue, maeValue, r2Value, matchedCount = ComputeMetrics(predictionCsvPath, CORRECTED_CSV)
    print(f"Fes={case.fes} 完成，RMSE={rmseValue:.6f}，MAE={maeValue:.6f}，R2={r2Value:.6f}")

    return {
        "fes": case.fes,
        "sample_count": int(matchedCount),
        "result_dir": str(resultInfo["resultDir"]),
        "prediction_csv": str(predictionCsvPath),
        "scatter_svg": str(resultInfo["scatterSvgPath"]) if resultInfo["scatterSvgPath"] else None,
        "visualization_svg": str(resultInfo["visualizationPath"]) if resultInfo["visualizationPath"] else None,
        "model_pkl": str(resultInfo["modelPath"]) if resultInfo["modelPath"] else None,
        "parameter_json": str(resultInfo["parameterPath"]) if resultInfo["parameterPath"] else None,
        "log_csv": str(resultInfo["logPath"]) if resultInfo["logPath"] else None,
        "used_high_csv": str(resultInfo["usedHighPath"]) if resultInfo["usedHighPath"] else None,
        "rmse": rmseValue,
        "mae": maeValue,
        "r2": r2Value,
        "status": "success",
    }


def Main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fesValues = list(range(15, 26))
    caseList = [SweepCase(fes=fesValue, caseDir=OUTPUT_ROOT / f"Fes{fesValue}") for fesValue in fesValues]

    resultRows = []
    failedRows = []

    for case in caseList:
        try:
            resultRows.append(RunOneCase(case))
        except Exception as exc:
            failedRows.append(
                {
                    "fes": case.fes,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"Fes={case.fes} 运行失败：{exc}")

    summaryColumns = [
        "fes",
        "sample_count",
        "result_dir",
        "prediction_csv",
        "scatter_svg",
        "visualization_svg",
        "model_pkl",
        "parameter_json",
        "log_csv",
        "used_high_csv",
        "rmse",
        "mae",
        "r2",
        "status",
    ]
    summaryFrame = pd.DataFrame(resultRows, columns=summaryColumns)
    if not summaryFrame.empty:
        summaryFrame = AddCompositeScores(summaryFrame)
        summaryFrame = summaryFrame.sort_values(["score_budget_balanced", "rmse", "fes"], ascending=[True, True, True]).reset_index(drop=True)
    summaryPath = OUTPUT_ROOT / "ma03_cl_fes_summary.csv"
    summaryFrame.to_csv(summaryPath, index=False, encoding="utf-8-sig")

    failureColumns = ["fes", "status", "error", "traceback"]
    failurePath = OUTPUT_ROOT / "ma03_cl_fes_failures.csv"
    pd.DataFrame(failedRows, columns=failureColumns).to_csv(failurePath, index=False, encoding="utf-8-sig")

    if summaryFrame.empty:
        print("没有成功结果。")
        return

    bestBalanced = summaryFrame.nsmallest(1, "score_budget_balanced").iloc[0].to_dict()
    bestAccuracy = summaryFrame.nsmallest(1, "rmse").iloc[0].to_dict()

    recommendation = {
        "recommended_policy": "优先兼顾 Fes 与误差",
        "recommended_result": bestBalanced,
        "best_accuracy_result": bestAccuracy,
        "fixed_model_config": BEST_MODEL_CONFIG,
        "data": {
            "initial_csv": str(INITIAL_CSV),
            "corrected_csv": str(CORRECTED_CSV),
        },
    }
    recommendationPath = OUTPUT_ROOT / "ma03_cl_best_recommendation.json"
    recommendationPath.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"扫参结束。成功 {len(summaryFrame)} 组，失败 {len(failedRows)} 组。")
    print(f"汇总文件：{summaryPath}")
    print(f"推荐结果：{recommendationPath}")
    print(f"综合推荐 Fes={int(bestBalanced['fes'])}，RMSE={float(bestBalanced['rmse']):.6f}，MAE={float(bestBalanced['mae']):.6f}，R2={float(bestBalanced['r2']):.6f}")
    print(f"纯精度最优 Fes={int(bestAccuracy['fes'])}，RMSE={float(bestAccuracy['rmse']):.6f}，MAE={float(bestAccuracy['mae']):.6f}，R2={float(bestAccuracy['r2']):.6f}")


if __name__ == "__main__":
    Main()
