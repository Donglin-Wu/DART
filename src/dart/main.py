"""
主入口。
"""

import os
import traceback

from .sequential import SequentialFusion


BEST_KRG_CORR = "matern52"
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


def Main():
    print("=== 双精度序贯融合程序 ===")

    cfdCsv = input("请输入低精度 CSV 文件路径: ").strip()
    if not os.path.exists(cfdCsv):
        print(f"错误：文件不存在 - {cfdCsv}")
        return

    fesText = input("请输入高精度样本上限 Fes: ").strip()
    try:
        fes = int(fesText)
    except Exception:
        print("错误：Fes 必须是整数。")
        return

    if fes < 6:
        print(f"错误：Fes 至少需要 6，当前为 {fes}。")
        return

    validationCsv = input("请输入全空间高精度验证 CSV 文件路径（可选，直接回车跳过）: ").strip()

    try:
        resultInfo = SequentialFusion(
            cfdCsv=cfdCsv,
            fes=fes,
            maxIters=50,
            randomState=42,
            krgCorr=BEST_KRG_CORR,
            validationCsv=validationCsv if validationCsv else None,
            modelConfig=BEST_MODEL_CONFIG,
            outputBaseDir=os.path.dirname(os.path.abspath(__file__)),
        )

        print("\n融合流程已完成。")
        print(f"结果目录: {resultInfo['resultDir']}")
        print(f"最终高精度样本数: {len(resultInfo['xHigh'])}")
        print(f"总迭代次数: {resultInfo['logDataFrame']['iteration'].max() if not resultInfo['logDataFrame'].empty else 0}")
        print("输出文件如下：")
        print(f"1. 已使用高精度样本: {resultInfo['usedHighPath']}")
        print(f"2. 运行日志: {resultInfo['logPath']}")
        print(f"3. 模型参数: {resultInfo['parameterPath']}")
        print(f"4. 融合模型: {resultInfo['modelPath']}")
        if resultInfo["visualizationPath"]:
            print(f"5. 融合可视化结果: {resultInfo['visualizationPath']}")
        print(f"6. 独立 pkl 预测结果 CSV: {resultInfo['predictionCsvPath']}")
        print(f"7. 回归拟合散点图: {resultInfo['scatterSvgPath']}")
    except Exception as exc:
        print(f"运行失败: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    Main()
