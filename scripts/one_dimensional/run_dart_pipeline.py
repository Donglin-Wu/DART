import builtins
import json
import os
import pickle
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data" / "raw" / "base_samples"
INITIAL_CSV = DATA_DIR / "Initial.csv"
CORRECTED_CSV = DATA_DIR / "Corrected.csv"
HIGH_CSV = DATA_DIR / "high.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dart import main, sequential  # noqa: E402
from dart.base import read_csv_auto  # noqa: E402
from dart.initialization import build_bounds_from_data, select_initial_points  # noqa: E402
from dart.kriging import create_kriging_model, fit_kriging, predict_kriging  # noqa: E402
from dart.optimizer import pso_search_extrema  # noqa: E402

RANDOM_STATE = 42
FES_GRID = [6, 7, 8, 9, 10, 11, 12, 14]
CORR_GRID = ["squar_exp", "matern32", "matern52"]

initial_df = pd.read_csv(INITIAL_CSV)
corrected_df = pd.read_csv(CORRECTED_CSV)
INPUT_COLS = list(initial_df.columns[:-1])
OUTPUT_COL = initial_df.columns[-1]


def row_key(values, ndigits=12):
    arr = np.asarray(values, dtype=float).ravel()
    return tuple(round(float(v), ndigits) for v in arr)


corrected_lookup = {
    row_key(row[INPUT_COLS].values): float(row[OUTPUT_COL])
    for _, row in corrected_df.iterrows()
}
available_coords = corrected_df[INPUT_COLS].values.astype(float)


def deterministic_lhs(bounds, n_samples, seed):
    dim = len(bounds)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    u = sampler.random(n=n_samples)
    l_bounds = np.array([b[0] for b in bounds], dtype=float)
    u_bounds = np.array([b[1] for b in bounds], dtype=float)
    return qmc.scale(u, l_bounds, u_bounds)


def low_scalers():
    X_low, y_low, _, _ = read_csv_auto(str(INITIAL_CSV))
    scaler_X_low = StandardScaler().fit(X_low)
    scaler_y_low = StandardScaler().fit(y_low.reshape(-1, 1))
    return X_low, y_low, scaler_X_low, scaler_y_low


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


def compute_initial_points(Fes, krg_corr, random_state):
    np.random.seed(random_state)
    X_low, y_low, scaler_X_low, scaler_y_low = low_scalers()
    dim = X_low.shape[1]
    N = sequential.initial_n(dim, Fes)

    X_low_s = scaler_X_low.transform(X_low)
    y_low_s = scaler_y_low.transform(y_low.reshape(-1, 1)).ravel()
    krg_L = create_kriging_model(
        dim=dim,
        random_state=random_state,
        poly_type="linear",
        nugget_val=1e-3,
        corr=krg_corr,
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

    snapped = [p for p in sequential.snap_targets_to_low_fidelity(initial_targets, available_coords) if p is not None]
    return sequential.fill_with_low_fidelity_points(snapped, available_coords, N)


def write_high_csv(points):
    rows = []
    for point in points:
        key = row_key(point)
        rows.append({**{col: float(v) for col, v in zip(INPUT_COLS, point)}, OUTPUT_COL: corrected_lookup[key]})
    pd.DataFrame(rows).to_csv(HIGH_CSV, index=False)


def auto_get_input(points, dim, is_initial=False, X_h=None, y_h=None):
    xs = []
    ys = []
    for point in points:
        if point is None:
            continue
        arr = np.asarray(point, dtype=float)
        key = row_key(arr)
        y_val = corrected_lookup[key]
        xs.append(arr)
        ys.append(y_val)
        print(f"[auto] {INPUT_COLS}={arr.tolist()} -> {OUTPUT_COL}={y_val}", flush=True)
    return sequential.to_2d_array(xs, dim), np.asarray(ys, dtype=float)


def render_svg_visualization(X_low, y_low, X_h, y_h, fusion_model,
                             scaler_X_low, scaler_y_low,
                             bounds, dim, iterations, Fes,
                             X_val=None, y_val=None,
                             input_labels=None, output_label="y"):
    if fusion_model is None or dim != 1:
        return

    x_min, x_max = bounds[0]
    x_plot = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    y_plot = predict_rows(fusion_model, x_plot, scaler_X_low, scaler_y_low)

    all_y = [*y_plot.tolist(), *np.asarray(y_low).tolist(), *np.asarray(y_h).tolist()]
    if X_val is not None and y_val is not None:
        all_y.extend(np.asarray(y_val).tolist())

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
    val_points = sorted(list(zip(X_val[:, 0], y_val)), key=lambda p: p[0]) if X_val is not None and y_val is not None else []

    xlabel = input_labels[0] if input_labels else "x"
    ylabel = output_label or "y"
    out_path = ROOT / "best_fusion_visualization.svg"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="24">Best fusion result (Fes={Fes}, iterations={iterations})</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black" stroke-width="2" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black" stroke-width="2" />
<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="18">{xlabel}</text>
<text x="25" y="{height/2}" transform="rotate(-90 25,{height/2})" text-anchor="middle" font-family="Arial" font-size="18">{ylabel}</text>
{polyline(fusion_points, "#228b22", 3)}
{polyline(val_points, "#111111", 2, 0.9) if val_points else ""}
{circles(low_points, "#4169e1", 4)}
{circles(high_points, "#dc143c", 5)}
</svg>"""
    out_path.write_text(svg, encoding="utf-8")
    print(f"[auto] saved visualization to {out_path}", flush=True)


def evaluate_fusion(fusion, scaler_X_low, scaler_y_low):
    X_val = corrected_df[INPUT_COLS].values.astype(float)
    y_val = corrected_df[OUTPUT_COL].values.astype(float)
    y_pred = predict_rows(fusion, X_val, scaler_X_low, scaler_y_low)
    rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_val)))
    return rmse, mae, y_pred


@contextmanager
def patched_runtime(krg_corr, random_state, final_run):
    orig_snap = sequential.snap_targets_to_low_fidelity
    orig_fill = sequential.fill_with_low_fidelity_points
    orig_get_input = sequential.get_input
    orig_visualize = sequential.visualize_results
    orig_lhs = sequential.latin_hypercube_sampling
    orig_input = builtins.input
    orig_main_seq = main.sequential_fusion

    def patched_snap(target_points, _X_low, used_points=None):
        return orig_snap(target_points, available_coords, used_points=used_points)

    def patched_fill(selected_points, _X_low, target_count):
        return orig_fill(selected_points, available_coords, target_count)

    def patched_lhs(bounds, n_samples):
        return deterministic_lhs(bounds, n_samples, seed=random_state + 17)

    def patched_main_seq(**kwargs):
        kwargs["krg_corr"] = krg_corr
        kwargs["random_state"] = random_state
        return sequential.sequential_fusion(**kwargs)

    sequential.snap_targets_to_low_fidelity = patched_snap
    sequential.fill_with_low_fidelity_points = patched_fill
    sequential.get_input = auto_get_input
    sequential.latin_hypercube_sampling = patched_lhs
    sequential.visualize_results = render_svg_visualization if final_run else (lambda *args, **kwargs: None)
    main.sequential_fusion = patched_main_seq

    try:
        yield
    finally:
        sequential.snap_targets_to_low_fidelity = orig_snap
        sequential.fill_with_low_fidelity_points = orig_fill
        sequential.get_input = orig_get_input
        sequential.visualize_results = orig_visualize
        sequential.latin_hypercube_sampling = orig_lhs
        builtins.input = orig_input
        main.sequential_fusion = orig_main_seq


def run_tuning_once(Fes, krg_corr, random_state):
    np.random.seed(random_state)
    X_low, y_low, scaler_X_low, scaler_y_low = low_scalers()

    with patched_runtime(krg_corr=krg_corr, random_state=random_state, final_run=False):
        initial_points = compute_initial_points(Fes=Fes, krg_corr=krg_corr, random_state=random_state)
        write_high_csv(initial_points)
        np.random.seed(random_state)

        answers = iter([str(HIGH_CSV)])
        builtins.input = lambda prompt="": next(answers)
        fusion, X_h, y_h, log_df = sequential.sequential_fusion(
            cfd_csv=str(INITIAL_CSV),
            Fes=Fes,
            max_iters=50,
            random_state=random_state,
            krg_corr=krg_corr,
            validation_csv=str(CORRECTED_CSV),
        )

    rmse, mae, _ = evaluate_fusion(fusion, scaler_X_low, scaler_y_low)
    return {
        "Fes": Fes,
        "krg_corr": krg_corr,
        "rmse": rmse,
        "mae": mae,
        "n_high": int(len(X_h)),
        "iterations": int(log_df["iteration"].max()) if not log_df.empty else 0,
        "initial_points": json.dumps(initial_points.ravel().tolist()),
    }


def find_new_outputs(before, after):
    return sorted(set(after) - set(before))


def run_best_via_main(best_cfg, random_state):
    np.random.seed(random_state)
    initial_points = None
    start_ts = time.time()

    with patched_runtime(krg_corr=best_cfg["krg_corr"], random_state=random_state, final_run=False):
        initial_points = compute_initial_points(
            Fes=int(best_cfg["Fes"]),
            krg_corr=best_cfg["krg_corr"],
            random_state=random_state,
        )
        write_high_csv(initial_points)
        np.random.seed(random_state)

        answers = iter([
            str(INITIAL_CSV),
            str(int(best_cfg["Fes"])),
            str(CORRECTED_CSV),
            str(HIGH_CSV),
        ])

        def fake_input(prompt=""):
            answer = next(answers)
            print(f"[auto-input] {prompt}{answer}", flush=True)
            return answer

        builtins.input = fake_input
        main.main()

    new_pkls = [p for p in ROOT.glob("trained_model_*.pkl") if p.stat().st_mtime >= start_ts - 1e-6]
    new_logs = [p for p in ROOT.glob("fusion_log_*.csv") if p.stat().st_mtime >= start_ts - 1e-6]
    new_params = [p for p in ROOT.glob("fusion_parameters_*.json") if p.stat().st_mtime >= start_ts - 1e-6]

    if not new_pkls:
        all_pkls = sorted(ROOT.glob("trained_model_*.pkl"), key=lambda p: p.stat().st_mtime)
        if not all_pkls:
            raise RuntimeError("no trained_model_*.pkl is available after the final run")
        model_path = all_pkls[-1]
    else:
        model_path = sorted(new_pkls, key=lambda p: p.stat().st_mtime)[-1]

    with model_path.open("rb") as f:
        model_data = pickle.load(f)

    fusion_model = model_data["fusion_model"]
    scaler_X_low = model_data["scaler_X_low"]
    scaler_y_low = model_data["scaler_y_low"]
    X_low, y_low, _, _ = read_csv_auto(str(INITIAL_CSV))
    used_hf_df = pd.read_csv(ROOT / "used_high_fidelity_data.csv")
    X_h = used_hf_df.iloc[:, :len(INPUT_COLS)].values.astype(float)
    y_h = used_hf_df.iloc[:, -1].values.astype(float)
    bounds = build_bounds_from_data(X_low)
    render_svg_visualization(
        X_low=X_low,
        y_low=y_low,
        X_h=X_h,
        y_h=y_h,
        fusion_model=fusion_model,
        scaler_X_low=scaler_X_low,
        scaler_y_low=scaler_y_low,
        bounds=bounds,
        dim=len(INPUT_COLS),
        iterations=int(best_cfg["iterations"]),
        Fes=int(best_cfg["Fes"]),
        X_val=corrected_df[INPUT_COLS].values.astype(float),
        y_val=corrected_df[OUTPUT_COL].values.astype(float),
        input_labels=INPUT_COLS,
        output_label=OUTPUT_COL,
    )

    rmse, mae, y_pred = evaluate_fusion(fusion_model, scaler_X_low, scaler_y_low)

    prediction_df = initial_df.copy()
    X_input = initial_df[INPUT_COLS].values.astype(float)
    prediction_df["fusion_prediction"] = predict_rows(fusion_model, X_input, scaler_X_low, scaler_y_low)
    prediction_csv = ROOT / "best_fusion_predictions.csv"
    prediction_df.to_csv(prediction_csv, index=False)

    summary = {
        "best_Fes": int(best_cfg["Fes"]),
        "best_krg_corr": best_cfg["krg_corr"],
        "rmse_on_corrected": rmse,
        "mae_on_corrected": mae,
        "initial_high_points": initial_points.ravel().tolist(),
        "trained_model_pkl": str(model_path),
        "fusion_log_csv": str(sorted(new_logs, key=lambda p: p.stat().st_mtime)[-1]) if new_logs else None,
        "fusion_parameters_json": str(sorted(new_params, key=lambda p: p.stat().st_mtime)[-1]) if new_params else None,
        "prediction_csv": str(prediction_csv),
        "visualization_svg": str(ROOT / "best_fusion_visualization.svg"),
        "high_csv": str(HIGH_CSV),
    }
    (ROOT / "dart_best_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main_pipeline():
    results = []
    for Fes in FES_GRID:
        for krg_corr in CORR_GRID:
            print(f"===== TUNE Fes={Fes}, krg_corr={krg_corr} =====", flush=True)
            try:
                result = run_tuning_once(Fes=Fes, krg_corr=krg_corr, random_state=RANDOM_STATE)
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as exc:
                error_result = {
                    "Fes": Fes,
                    "krg_corr": krg_corr,
                    "rmse": None,
                    "mae": None,
                    "n_high": None,
                    "iterations": None,
                    "initial_points": None,
                    "error": repr(exc),
                }
                results.append(error_result)
                print(json.dumps(error_result, ensure_ascii=False), flush=True)

    results_df = pd.DataFrame(results)
    valid_results_df = results_df.dropna(subset=["rmse", "mae"]).sort_values(
        ["rmse", "mae", "Fes"], ascending=[True, True, True]
    )
    tuning_csv = ROOT / "dart_tuning_results.csv"
    results_df.to_csv(tuning_csv, index=False)

    if valid_results_df.empty:
        raise RuntimeError("all tuning runs failed; no valid result available")

    best_cfg = valid_results_df.iloc[0].to_dict()
    summary = run_best_via_main(best_cfg=best_cfg, random_state=RANDOM_STATE)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main_pipeline()
