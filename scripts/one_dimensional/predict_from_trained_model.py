import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd


def predict_rows(fusion_model, X, scaler_X_low, scaler_y_low):
    preds = []
    for row in X:
        pred = fusion_model.predict(
            row.reshape(1, -1),
            scaler_X_low=scaler_X_low,
            scaler_y_low=scaler_y_low,
        )
        preds.append(float(pred.ravel()[0]))
    return preds


def main():
    parser = argparse.ArgumentParser(description="Generate fusion predictions from a trained DART PKL model.")
    parser.add_argument("--model", required=True, help="Path to trained_model_*.pkl")
    parser.add_argument("--input", required=True, help="Path to the input CSV")
    parser.add_argument("--output", required=True, help="Path to the output CSV")
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent
    code_root = script_root.parents[1]
    src_dir = code_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    with model_path.open("rb") as f:
        model_data = pickle.load(f)

    fusion_model = model_data["fusion_model"]
    scaler_X_low = model_data["scaler_X_low"]
    scaler_y_low = model_data["scaler_y_low"]
    dim = int(model_data["dim"])

    df = pd.read_csv(input_path)
    X = df.iloc[:, :dim].values.astype(float)
    y_pred = predict_rows(fusion_model, X, scaler_X_low, scaler_y_low)

    out_df = df.copy()
    out_df["fusion_prediction"] = y_pred
    out_df.to_csv(output_path, index=False)
    print(f"saved predictions to {output_path}")


if __name__ == "__main__":
    main()
