import os
import builtins
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / 'src'
DATA = ROOT / 'data' / 'raw' / 'base_samples'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dart import sequential
from dart.base import read_csv_auto
from dart.initialization import build_bounds_from_data, latin_hypercube_sampling, select_initial_points
from dart.kriging import create_kriging_model, fit_kriging, predict_kriging
from dart.optimizer import pso_search_extrema
from sklearn.preprocessing import StandardScaler

INITIAL_CSV = DATA / 'Initial.csv'
CORRECTED_CSV = DATA / 'Corrected.csv'
HIGH_CSV = DATA / 'high.csv'
FES = 11
KRIG_CORR = 'squar_exp'

corrected_df = pd.read_csv(CORRECTED_CSV)
corrected_map = {round(float(row['Alpha']), 12): float(row['CL']) for _, row in corrected_df.iterrows()}
available_coords = np.array(sorted(corrected_map.keys()), dtype=float).reshape(-1, 1)

orig_snap = sequential.snap_targets_to_low_fidelity
orig_fill = sequential.fill_with_low_fidelity_points
orig_get_input = sequential.get_input
orig_input = builtins.input
orig_visualize = sequential.visualize_results


def alpha_key(point):
    return round(float(np.asarray(point, dtype=float).ravel()[0]), 12)


def patched_snap_targets_to_low_fidelity(target_points, _X_low, used_points=None):
    return orig_snap(target_points, available_coords, used_points=used_points)


def patched_fill_with_low_fidelity_points(selected_points, _X_low, target_count):
    return orig_fill(selected_points, available_coords, target_count)


def auto_get_input(points, dim, is_initial=False, X_h=None, y_h=None):
    xs = []
    ys = []
    for point in points:
        if point is None:
            continue
        key = alpha_key(point)
        xs.append(np.asarray(point, dtype=float))
        ys.append(corrected_map[key])
        print(f'[auto-final] Alpha={key}, CL={corrected_map[key]}')
    return sequential.to_2d_array(xs, dim), np.asarray(ys, dtype=float)


def compute_initial_points():
    X_low, y_low, _, _ = read_csv_auto(str(INITIAL_CSV))
    dim = X_low.shape[1]
    N = sequential.initial_n(dim, FES)

    scaler_X_low = StandardScaler().fit(X_low)
    scaler_y_low = StandardScaler().fit(y_low.reshape(-1, 1))
    X_low_s = scaler_X_low.transform(X_low)
    y_low_s = scaler_y_low.transform(y_low.reshape(-1, 1)).ravel()

    krg_L = create_kriging_model(dim=dim, random_state=42, poly_type='linear', nugget_val=1e-3, corr=KRIG_CORR)
    krg_L = fit_kriging(krg_L, X_low_s, y_low_s, normalize=True)
    bounds = build_bounds_from_data(X_low)

    def low_fidelity_func(x):
        x_scaled = scaler_X_low.transform(np.asarray(x, dtype=float).reshape(1, -1))
        mu = predict_kriging(krg_L, x_scaled)
        mu_orig = scaler_y_low.inverse_transform(mu.reshape(-1, 1)).ravel()
        return mu_orig[0]

    min_points, max_points = pso_search_extrema(low_fidelity_func, bounds, n_searches=2 * dim, pop_size=5 * dim, iters=30)
    selected_extrema = select_initial_points(min_points, max_points)
    current_points_list = selected_extrema.tolist() if len(selected_extrema) > 0 else []

    for d in range(dim):
        lb = bounds[d][0]
        ub = bounds[d][1]
        span = ub - lb
        tol = 0.05 * span
        has_lb = len(current_points_list) > 0 and np.min(np.abs(np.array(current_points_list)[:, d] - lb)) < tol
        if not has_lb:
            new_pt = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
            new_pt[d] = lb
            current_points_list.append(new_pt)
        has_ub = len(current_points_list) > 0 and np.min(np.abs(np.array(current_points_list)[:, d] - ub)) < tol
        if not has_ub:
            new_pt = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
            new_pt[d] = ub
            current_points_list.append(new_pt)

    initial_targets = sequential.to_2d_array(current_points_list, dim)
    if len(initial_targets) < N:
        n_needed = N - len(initial_targets)
        candidate_pool = latin_hypercube_sampling(bounds, max(n_needed * 100, 1000))
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

    snapped = [p for p in patched_snap_targets_to_low_fidelity(initial_targets, X_low) if p is not None]
    return patched_fill_with_low_fidelity_points(snapped, X_low, N)


def write_svg(fusion, scaler_X_low, scaler_y_low, X_low, y_low, X_h, y_h):
    print('[marker] write_svg start', flush=True)
    bounds = [(-5.0, 25.0)]
    x_min, x_max = bounds[0]
    x_plot = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    y_plot = fusion.predict(x_plot, scaler_X_low=scaler_X_low, scaler_y_low=scaler_y_low)
    print('[marker] write_svg predicted grid', flush=True)
    X_val = corrected_df[['Alpha']].values.astype(float)
    y_val = corrected_df['CL'].values.astype(float)

    all_y = [*y_plot.tolist(), *y_low.tolist(), *y_h.tolist(), *y_val.tolist()]
    y_min = float(min(all_y))
    y_max = float(max(all_y))
    width, height = 1200, 800
    left, right, top, bottom = 90, 40, 60, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x):
        return left + (float(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return top + (y_max - float(y)) / (y_max - y_min) * plot_h

    def polyline(points, color, stroke_width=2, opacity=1.0):
        pts = ' '.join(f'{sx(x)},{sy(y)}' for x, y in points)
        return f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}" points="{pts}" />'

    def circles(points, color, radius):
        return '\n'.join(
            f'<circle cx="{sx(x)}" cy="{sy(y)}" r="{radius}" fill="{color}" opacity="0.85" />'
            for x, y in points
        )

    fusion_points = list(zip(x_plot[:, 0], y_plot))
    low_points = list(zip(X_low[:, 0], y_low))
    high_points = list(zip(X_h[:, 0], y_h))
    val_points = sorted(list(zip(X_val[:, 0], y_val)), key=lambda p: p[0])

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="24">Fusion result (Fes={FES})</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black" stroke-width="2" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black" stroke-width="2" />
{text_label('Alpha', width/2, height-20)}
<text x="25" y="{height/2}" transform="rotate(-90 25,{height/2})" text-anchor="middle" font-family="Arial" font-size="18">CL</text>
{polyline(fusion_points, '#228b22', 3)}
{polyline(val_points, '#111111', 2, 0.9)}
{circles(low_points, '#4169e1', 4)}
{circles(high_points, '#dc143c', 5)}
</svg>'''
    print('[marker] write_svg built string', flush=True)
    out_path = ROOT / 'fusion_visualization_1d_Fes11.svg'
    out_path.write_text(svg, encoding='utf-8')
    print('[marker] write_svg wrote file', flush=True)
    return out_path


def text_label(text, x, y):
    return f'<text x="{x}" y="{y}" text-anchor="middle" font-family="Arial" font-size="18">{text}</text>'

try:
    sequential.snap_targets_to_low_fidelity = patched_snap_targets_to_low_fidelity
    sequential.fill_with_low_fidelity_points = patched_fill_with_low_fidelity_points
    sequential.get_input = auto_get_input
    sequential.visualize_results = lambda *args, **kwargs: None

    initial_points = compute_initial_points()
    pd.DataFrame([{'Alpha': alpha_key(p), 'CL': corrected_map[alpha_key(p)]} for p in initial_points]).to_csv(HIGH_CSV, index=False)

    builtins.input = lambda prompt='': str(HIGH_CSV)
    fusion, X_h, y_h, log_df = sequential.sequential_fusion(
        cfd_csv=str(INITIAL_CSV),
        Fes=FES,
        max_iters=50,
        random_state=42,
        krg_corr=KRIG_CORR,
        validation_csv=str(CORRECTED_CSV),
    )
    print('[marker] sequential_fusion returned', flush=True)

    X_low, y_low, _, _ = read_csv_auto(str(INITIAL_CSV))
    print('[marker] low data reloaded', flush=True)
    scaler_X_low = StandardScaler().fit(X_low)
    scaler_y_low = StandardScaler().fit(y_low.reshape(-1, 1))
    print('[marker] scalers ready', flush=True)
    X_val = corrected_df[['Alpha']].values.astype(float)
    y_val = corrected_df['CL'].values.astype(float)
    y_pred = fusion.predict(X_val, scaler_X_low=scaler_X_low, scaler_y_low=scaler_y_low)
    print('[marker] prediction done', flush=True)
    rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_val)))
    print('[marker] metrics done', flush=True)
    svg_path = write_svg(fusion, scaler_X_low, scaler_y_low, X_low, y_low, X_h, y_h)
    print('[marker] svg done', flush=True)

    summary = {
        'Fes': FES,
        'rmse': rmse,
        'mae': mae,
        'n_high': int(len(X_h)),
        'iterations': int(log_df['iteration'].max()) if not log_df.empty else 0,
        'initial_points': initial_points.ravel().tolist(),
        'all_high_points': X_h.ravel().tolist(),
        'svg': str(svg_path),
    }
    Path(ROOT / 'dart_final_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
finally:
    builtins.input = orig_input
    sequential.snap_targets_to_low_fidelity = orig_snap
    sequential.fill_with_low_fidelity_points = orig_fill
    sequential.get_input = orig_get_input
    sequential.visualize_results = orig_visualize
