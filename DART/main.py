"""
Main entry point.
"""

import os
import traceback

from sequential import SequentialFusion


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
    print("=== Dual-fidelity Sequential Fusion Program ===")

    cfdCsv = input("Enter the low-fidelity CSV file path: ").strip()
    if not os.path.exists(cfdCsv):
        print(f"Error: file does not exist - {cfdCsv}")
        return

    bhfText = input("Enter the high-fidelity budget limit BHF: ").strip()
    try:
        bhf = int(bhfText)
    except Exception:
        print("Error: BHF must be an integer.")
        return

    if bhf < 6:
        print(f"Error: BHF must be at least 6; current value is {bhf}.")
        return

    validationCsv = input("Enter the full-space high-fidelity validation CSV path (optional, press Enter to skip): ").strip()

    try:
        resultInfo = SequentialFusion(
            cfdCsv=cfdCsv,
            bhf=bhf,
            maxIters=50,
            randomState=42,
            krgCorr=BEST_KRG_CORR,
            validationCsv=validationCsv if validationCsv else None,
            modelConfig=BEST_MODEL_CONFIG,
            outputBaseDir=os.path.dirname(os.path.abspath(__file__)),
        )

        print("\nFusion workflow completed.")
        print(f"Result directory: {resultInfo['resultDir']}")
        print(f"Final high-fidelity sample count: {len(resultInfo['xHigh'])}")
        print(f"Total iterations: {resultInfo['logDataFrame']['iteration'].max() if not resultInfo['logDataFrame'].empty else 0}")
        print("Output files:")
        print(f"1. Used high-fidelity samples: {resultInfo['usedHighPath']}")
        print(f"2. Run log: {resultInfo['logPath']}")
        print(f"3. Model parameters: {resultInfo['parameterPath']}")
        print(f"4. Fusion model: {resultInfo['modelPath']}")
        if resultInfo["visualizationPath"]:
            print(f"5. Fusion visualization result: {resultInfo['visualizationPath']}")
        print(f"6. Standalone pkl prediction CSV: {resultInfo['predictionCsvPath']}")
        print(f"7. Regression-fit scatter plot: {resultInfo['scatterSvgPath']}")
    except Exception as exc:
        print(f"Run failed: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    Main()
