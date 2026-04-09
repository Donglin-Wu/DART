import builtins
import json
import os
import pickle
import sys
import time
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLBACKEND", "Agg")

RUN_DIR = Path(__file__).resolve().parent
CODE_DIR = RUN_DIR.parents[1]
SRC_DIR = CODE_DIR / "src"
DATA_DIR = CODE_DIR / "data" / "raw" / "base_samples"
INITIAL_CSV = DATA_DIR / "Initial.csv"
CORRECTED_CSV = DATA_DIR / "Corrected.csv"
HIGH_CSV = DATA_DIR / "high.csv"
RANDOM_STATE = 42

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dart import sequential  # noqa: E402
from dart.base import read_csv_auto  # noqa: E402
from dart.initialization import build_bounds_from_data, select_initial_points  # noqa: E402
from dart.kriging import create_kriging_model, fit_kriging, predict_kriging  # noqa: E402
from dart.optimizer import pso_search_extrema  # noqa: E402

INPUT_DF = pd.read_csv(INITIAL_CSV)
CORRECTED_DF = pd.read_csv(CORRECTED_CSV)
INPUT_COLS = list(INPUT_DF.columns[:-1])
OUTPUT_COL = INPUT_DF.columns[-1]


def row_key(values, ndigits=12):
    arr = np.asarray(values, dtype=float).ravel()
    return tuple(round(float(v), ndigits) for v in arr)


CORRECTED_LOOKUP = {
    row_key(row[INPUT_COLS].values): float(row[OUTPUT_COL])
    for _, row in CORRECTED_DF.iterrows()
}
AVAILABLE_COORDS = CORRECTED_DF[INPUT_COLS].values.astype(float)


def deterministic_lhs(bounds, n_samples, seed):
    dim = len(bounds)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    u = sampler.random(n=n_samples)
    l_bounds = np.array([b[0] for b in bounds], dtype=float)
    u_bounds = np.array([b[1] for b in bounds], dtype=float)
    return qmc.scale(u, l_bounds, u_bounds)


def predict_rows(fusion_model, X, scaler_X_low, scaler_y_low):
    X = np.asarray(X, dtype=float)
    preds = []
    for row in X:
        pred = fusion_model.predict(
            np.asarray(row, dtype=float).reshape(1, -1),
            scaler_X_low=scaler_X_low,
            scaler_y_low=scaler_y_low,
        )
        preds.append(float(np.asarray(pred).ravel()[0]))
    return np.asarray(preds, dtype=float)


def low_scalers():
    X_low, y_low, _, _ = read_csv_auto(str(INITIAL_CSV))
    scaler_X_low = StandardScaler().fit(X_low)
    scaler_y_low = StandardScaler().fit(y_low.reshape(-1, 1))
    return X_low, y_low, scaler_X_low, scaler_y_low


def build_model_config(krg_corr, low_poly_type, delta_poly_type, low_nugget, delta_nugget, l_min, l_max):
    return {
        "low": {
            "corr": krg_corr,
            "poly_type": low_poly_type,
            "nugget_val": low_nugget,
            "l_min": l_min,
            "l_max": l_max,
        },
        "delta": {
            "corr": krg_corr,
            "poly_type": delta_poly_type,
            "nugget_val": delta_nugget,
            "l_min": l_min,
            "l_max": l_max,
        },
    }


def compute_initial_points(Fes, model_config, random_state):
    np.random.seed(random_state)
    X_low, y_low, scaler_X_low, scaler_y_low = low_scalers()
    dim = X_low.shape[1]
    N = sequential.initial_n(dim, Fes)

    X_low_s = scaler_X_low.transform(X_low)
    y_low_s = scaler_y_low.transform(y_low.reshape(-1, 1)).ravel()
    low_cfg = model_config["low"]
    krg_L = create_kriging_model(
        dim=dim,
        random_state=random_state,
        poly_type=low_cfg["poly_type"],
        nugget_val=low_cfg["nugget_val"],
        corr=low_cfg["corr"],
        l_min=low_cfg["l_min"],
        l_max=low_cfg["l_max"],
    )
    krg_L = fit_kriging(krg_L, X_low_s, y_low_s, normalize=True)
    bounds = build_bounds_from_data(X_low)

    def low_fidelity_func(x):
        x_scaled = scaler_X_low.transform(np.asarray(x, dtype=float).reshape(1, -1))
        mu = predict_kriging(krg_L, x_scaled)
        mu_orig = scaler_y_low.inverse_transform(mu.reshape(-1, 1)).ravel()
        return mu_orig[0]

    min_points, max_points = pso_search_extrema(
        model_func=low_fidelity_func,
        bounds=bounds,
        n_searches=2 * dim,
        pop_size=5 * dim,
        iters=30,
        random_state=random_state,
    )
    selected_extrema = select_initial_points(min_points, max_points)
    current_points_list = selected_extrema.tolist() if len(selected_extrema) > 0 else []

    for d in range(dim):
        lb = bounds[d][0]
        ub = bounds[d][1]
        span = ub - lb
        tol = 0.05 * span

        has_lb = False
        if len(current_points_list) > 0:
            existing_arr = np.array(current_points_list)
            if np.min(np.abs(existing_arr[:, d] - lb)) < tol:
                has_lb = True

        if not has_lb:
            new_pt = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
            new_pt[d] = lb
            current_points_list.append(new_pt)

        has_ub = False
        if len(current_points_list) > 0:
            existing_arr = np.array(current_points_list)
            if np.min(np.abs(existing_arr[:, d] - ub)) < tol:
                has_ub = True

        if not has_ub:
            new_pt = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
            new_pt[d] = ub
            current_points_list.append(new_pt)

    initial_targets = sequential.to_2d_array(current_points_list, dim)

    if len(initial_targets) < N:
        n_needed = N - len(initial_targets)
        candidate_pool = deterministic_lhs(bounds, max(n_needed * 100, 1000), seed=random_state + 17)
        for _ in range(n_needed):
            if len(initial_targets) == 0:
                best_cand = candidate_pool[0]
            else:
                best_cand = None
                max_min_dist = -1.0
                np.random.shuffle(candidate_pool)
                for cand in candidate_pool[:500]:
                    dists = np.linalg.norm(initial_targets - cand, axis=1)
                    min_d = np.min(dists)
                    if min_d > max_min_dist:
                        max_min_dist = min_d
                        best_cand = cand
            initial_targets = np.vstack([initial_targets, best_cand])

    snapped = [
        point for point in sequential.snap_targets_to_low_fidelity(initial_targets, AVAILABLE_COORDS) if point is not None
    ]
    return sequential.fill_with_low_fidelity_points(snapped, AVAILABLE_COORDS, N)


def write_high_csv(points):
    rows = []
    for point in points:
        key = row_key(point)
        rows.append({**{col: float(v) for col, v in zip(INPUT_COLS, point)}, OUTPUT_COL: CORRECTED_LOOKUP[key]})
    pd.DataFrame(rows).to_csv(HIGH_CSV, index=False)


def auto_get_input(points, dim, is_initial=False, X_h=None, y_h=None):
    xs = []
    ys = []
    for point in points:
        if point is None:
            continue
        arr = np.asarray(point, dtype=float)
        key = row_key(arr)
        y_val = CORRECTED_LOOKUP[key]
        xs.append(arr)
        ys.append(y_val)
    return sequential.to_2d_array(xs, dim), np.asarray(ys, dtype=float)


@contextmanager
def patched_runtime(random_state):
    orig_snap = sequential.snap_targets_to_low_fidelity
    orig_fill = sequential.fill_with_low_fidelity_points
    orig_get_input = sequential.get_input
    orig_visualize = sequential.visualize_results
    orig_lhs = sequential.latin_hypercube_sampling
    orig_input = builtins.input

    def patched_snap(target_points, _X_low, used_points=None):
        return orig_snap(target_points, AVAILABLE_COORDS, used_points=used_points)

    def patched_fill(selected_points, _X_low, target_count):
        return orig_fill(selected_points, AVAILABLE_COORDS, target_count)

    def patched_lhs(bounds, n_samples):
        return deterministic_lhs(bounds, n_samples, seed=random_state + 17)

    sequential.snap_targets_to_low_fidelity = patched_snap
    sequential.fill_with_low_fidelity_points = patched_fill
    sequential.get_input = auto_get_input
    sequential.visualize_results = lambda *args, **kwargs: None
    sequential.latin_hypercube_sampling = patched_lhs

    try:
        yield
    finally:
        sequential.snap_targets_to_low_fidelity = orig_snap
        sequential.fill_with_low_fidelity_points = orig_fill
        sequential.get_input = orig_get_input
        sequential.visualize_results = orig_visualize
        sequential.latin_hypercube_sampling = orig_lhs
        builtins.input = orig_input


def evaluate_fusion(fusion_model, scaler_X_low, scaler_y_low):
    X_val = CORRECTED_DF[INPUT_COLS].values.astype(float)
    y_val = CORRECTED_DF[OUTPUT_COL].values.astype(float)
    y_pred = predict_rows(fusion_model, X_val, scaler_X_low, scaler_y_low)
    rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_val)))
    return rmse, mae


def latest_file(pattern, start_ts):
    candidates = [p for p in RUN_DIR.glob(pattern) if p.stat().st_mtime >= start_ts - 1e-6]
    if candidates:
        return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    all_candidates = list(RUN_DIR.glob(pattern))
    if not all_candidates:
        return None
    return sorted(all_candidates, key=lambda p: p.stat().st_mtime)[-1]


def render_svg_visualization(model_path, Fes, iterations):
    with model_path.open("rb") as f:
        model_data = pickle.load(f)

    fusion_model = model_data["fusion_model"]
    scaler_X_low = model_data["scaler_X_low"]
    scaler_y_low = model_data["scaler_y_low"]
    X_low, y_low, _, _ = read_csv_auto(str(INITIAL_CSV))
    used_hf_df = pd.read_csv(RUN_DIR / "used_high_fidelity_data.csv")
    X_h = used_hf_df.iloc[:, :len(INPUT_COLS)].values.astype(float)
    y_h = used_hf_df.iloc[:, -1].values.astype(float)

    bounds = build_bounds_from_data(X_low)
    x_min, x_max = bounds[0]
    x_plot = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    y_plot = predict_rows(fusion_model, x_plot, scaler_X_low, scaler_y_low)

    all_y = [*y_plot.tolist(), *np.asarray(y_low).tolist(), *np.asarray(y_h).tolist(), *CORRECTED_DF[OUTPUT_COL].tolist()]
    y_min = float(min(all_y))
    y_max = float(max(all_y))
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    width, height = 1200, 800
    left, right, top, bottom = 90, 40, 60, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x):
        return left + (float(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return top + (y_max - float(y)) / (y_max - y_min) * plot_h

    def polyline(points, color, stroke_width=2, opacity=1.0):
        pts = " ".join(f"{sx(x)},{sy(y)}" for x, y in points)
        return f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}" points="{pts}" />'

    def circles(points, color, radius):
        return "\n".join(
            f'<circle cx="{sx(x)}" cy="{sy(y)}" r="{radius}" fill="{color}" opacity="0.85" />'
            for x, y in points
        )

    fusion_points = list(zip(x_plot[:, 0], y_plot))
    low_points = list(zip(X_low[:, 0], y_low))
    high_points = list(zip(X_h[:, 0], y_h))
    val_points = sorted(list(zip(CORRECTED_DF[INPUT_COLS[0]].tolist(), CORRECTED_DF[OUTPUT_COL].tolist())), key=lambda p: p[0])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="24">Extended best fusion result (Fes={Fes}, iterations={iterations})</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black" stroke-width="2" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black" stroke-width="2" />
<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="18">{INPUT_COLS[0]}</text>
<text x="25" y="{height/2}" transform="rotate(-90 25,{height/2})" text-anchor="middle" font-family="Arial" font-size="18">{OUTPUT_COL}</text>
{polyline(fusion_points, "#228b22", 3)}
{polyline(val_points, "#111111", 2, 0.9)}
{circles(low_points, "#4169e1", 4)}
{circles(high_points, "#dc143c", 5)}
</svg>"""
    out_path = RUN_DIR / "extended_best_visualization.svg"
    out_path.write_text(svg, encoding="utf-8")
    return out_path


def run_once(Fes, model_config, random_state):
    os.chdir(RUN_DIR)
    np.random.seed(random_state)
    X_low, y_low, scaler_X_low, scaler_y_low = low_scalers()

    with patched_runtime(random_state=random_state):
        initial_points = compute_initial_points(Fes=Fes, model_config=model_config, random_state=random_state)
        write_high_csv(initial_points)
        np.random.seed(random_state)

        answers = iter([str(HIGH_CSV)])
        builtins.input = lambda prompt="": next(answers)

        start_ts = time.time()
        fusion_model, X_h, y_h, log_df = sequential.sequential_fusion(
            cfd_csv=str(INITIAL_CSV),
            Fes=Fes,
            max_iters=50,
            random_state=random_state,
            krg_corr=model_config["low"]["corr"],
            validation_csv=str(CORRECTED_CSV),
            model_config=model_config,
        )

    rmse, mae = evaluate_fusion(fusion_model, scaler_X_low, scaler_y_low)
    result = {
        "Fes": Fes,
        "krg_corr": model_config["low"]["corr"],
        "low_poly_type": model_config["low"]["poly_type"],
        "delta_poly_type": model_config["delta"]["poly_type"],
        "low_nugget": model_config["low"]["nugget_val"],
        "delta_nugget": model_config["delta"]["nugget_val"],
        "l_min": model_config["low"]["l_min"],
        "l_max": model_config["low"]["l_max"],
        "rmse": rmse,
        "mae": mae,
        "n_high": int(len(X_h)),
        "iterations": int(log_df["iteration"].max()) if not log_df.empty else 0,
        "initial_points": json.dumps(initial_points.ravel().tolist()),
        "trained_model_pkl": str(latest_file("trained_model_*.pkl", start_ts)),
        "fusion_log_csv": str(latest_file("fusion_log_*.csv", start_ts)),
        "fusion_parameters_json": str(latest_file("fusion_parameters_*.json", start_ts)),
    }
    return result


def save_predictions_from_pkl(model_path, output_path):
    with model_path.open("rb") as f:
        model_data = pickle.load(f)
    fusion_model = model_data["fusion_model"]
    scaler_X_low = model_data["scaler_X_low"]
    scaler_y_low = model_data["scaler_y_low"]

    out_df = INPUT_DF.copy()
    X_input = INPUT_DF[INPUT_COLS].values.astype(float)
    out_df["fusion_prediction"] = predict_rows(fusion_model, X_input, scaler_X_low, scaler_y_low)
    out_df.to_csv(output_path, index=False)


def main():
    os.chdir(RUN_DIR)

    search_space = {
        "Fes": [14, 16, 18],
        "krg_corr": ["matern52"],
        "low_poly_type": ["linear", "constant"],
        "delta_poly_type": ["constant", "linear"],
        "low_nugget": [1e-4, 1e-3, 1e-2],
        "delta_nugget": [1e-8, 1e-6],
        "length_bounds": [(0.25, 10.0), (0.5, 20.0), (1.0, 30.0)],
    }

    results = []
    for Fes, krg_corr, low_poly, delta_poly, low_nugget, delta_nugget, (l_min, l_max) in product(
        search_space["Fes"],
        search_space["krg_corr"],
        search_space["low_poly_type"],
        search_space["delta_poly_type"],
        search_space["low_nugget"],
        search_space["delta_nugget"],
        search_space["length_bounds"],
    ):
        model_config = build_model_config(
            krg_corr=krg_corr,
            low_poly_type=low_poly,
            delta_poly_type=delta_poly,
            low_nugget=low_nugget,
            delta_nugget=delta_nugget,
            l_min=l_min,
            l_max=l_max,
        )
        print(
            f"RUN Fes={Fes}, corr={krg_corr}, low_poly={low_poly}, delta_poly={delta_poly}, "
            f"low_nugget={low_nugget}, delta_nugget={delta_nugget}, l_bounds=({l_min}, {l_max})",
            flush=True,
        )
        try:
            result = run_once(Fes=Fes, model_config=model_config, random_state=RANDOM_STATE)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            result = {
                "Fes": Fes,
                "krg_corr": krg_corr,
                "low_poly_type": low_poly,
                "delta_poly_type": delta_poly,
                "low_nugget": low_nugget,
                "delta_nugget": delta_nugget,
                "l_min": l_min,
                "l_max": l_max,
                "rmse": None,
                "mae": None,
                "n_high": None,
                "iterations": None,
                "initial_points": None,
                "trained_model_pkl": None,
                "fusion_log_csv": None,
                "fusion_parameters_json": None,
                "error": repr(exc),
            }
            print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_csv = RUN_DIR / "extended_tuning_results.csv"
    results_df.to_csv(results_csv, index=False)

    valid_results = results_df.dropna(subset=["rmse", "mae"]).sort_values(["rmse", "mae", "Fes"], ascending=[True, True, True])
    if valid_results.empty:
        raise RuntimeError("no valid tuning result was produced")

    best = valid_results.iloc[0].to_dict()
    best_model_path = Path(best["trained_model_pkl"])
    best_prediction_csv = RUN_DIR / "extended_best_predictions.csv"
    save_predictions_from_pkl(best_model_path, best_prediction_csv)
    best_svg = render_svg_visualization(best_model_path, int(best["Fes"]), int(best["iterations"]))

    summary = {
        "search_space": {
            "Fes": search_space["Fes"],
            "krg_corr": search_space["krg_corr"],
            "low_poly_type": search_space["low_poly_type"],
            "delta_poly_type": search_space["delta_poly_type"],
            "low_nugget": search_space["low_nugget"],
            "delta_nugget": search_space["delta_nugget"],
            "length_bounds": search_space["length_bounds"],
        },
        "best_parameters": {
            "Fes": int(best["Fes"]),
            "krg_corr": best["krg_corr"],
            "low_poly_type": best["low_poly_type"],
            "delta_poly_type": best["delta_poly_type"],
            "low_nugget": float(best["low_nugget"]),
            "delta_nugget": float(best["delta_nugget"]),
            "l_min": float(best["l_min"]),
            "l_max": float(best["l_max"]),
        },
        "metrics": {
            "rmse_on_corrected": float(best["rmse"]),
            "mae_on_corrected": float(best["mae"]),
        },
        "outputs": {
            "results_csv": str(results_csv),
            "best_model_pkl": str(best_model_path),
            "best_log_csv": best["fusion_log_csv"],
            "best_parameters_json": best["fusion_parameters_json"],
            "best_prediction_csv": str(best_prediction_csv),
            "best_visualization_svg": str(best_svg),
            "high_csv": str(HIGH_CSV),
        },
    }
    summary_path = RUN_DIR / "extended_best_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
