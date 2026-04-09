import os
import builtins
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ['MPLBACKEND'] = 'Agg'

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / 'src'
DATA = ROOT / 'data' / 'raw' / 'base_samples'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dart import main, sequential
from dart.base import read_csv_auto
from dart.initialization import build_bounds_from_data, latin_hypercube_sampling, select_initial_points
from dart.kriging import create_kriging_model, fit_kriging, predict_kriging
from dart.optimizer import pso_search_extrema
from sklearn.preprocessing import StandardScaler

INITIAL_CSV = DATA / 'Initial.csv'
CORRECTED_CSV = DATA / 'Corrected.csv'
HIGH_CSV = DATA / 'high.csv'

corrected_df = pd.read_csv(CORRECTED_CSV)
corrected_map = {round(float(row['Alpha']), 12): float(row['CL']) for _, row in corrected_df.iterrows()}
available_coords = np.array(sorted(corrected_map.keys()), dtype=float).reshape(-1, 1)

orig_snap = sequential.snap_targets_to_low_fidelity
orig_fill = sequential.fill_with_low_fidelity_points
orig_get_input = sequential.get_input
orig_input = builtins.input
orig_visualize = sequential.visualize_results


def patched_snap_targets_to_low_fidelity(target_points, _X_low, used_points=None):
    return orig_snap(target_points, available_coords, used_points=used_points)


def patched_fill_with_low_fidelity_points(selected_points, _X_low, target_count):
    return orig_fill(selected_points, available_coords, target_count)


def alpha_key(point):
    return round(float(np.asarray(point, dtype=float).ravel()[0]), 12)


def compute_initial_points(Fes, random_state=42, krg_corr='squar_exp'):
    X_low, y_low, _, _ = read_csv_auto(str(INITIAL_CSV))
    dim = X_low.shape[1]
    N = sequential.initial_n(dim, Fes)

    scaler_X_low = StandardScaler().fit(X_low)
    scaler_y_low = StandardScaler().fit(y_low.reshape(-1, 1))
    X_low_s = scaler_X_low.transform(X_low)
    y_low_s = scaler_y_low.transform(y_low.reshape(-1, 1)).ravel()

    krg_L = create_kriging_model(dim=dim, random_state=random_state, poly_type='linear', nugget_val=1e-3, corr=krg_corr)
    krg_L = fit_kriging(krg_L, X_low_s, y_low_s, normalize=True)
    bounds = build_bounds_from_data(X_low)

    def low_fidelity_func(x):
        x_scaled = scaler_X_low.transform(np.asarray(x, dtype=float).reshape(1, -1))
        mu = predict_kriging(krg_L, x_scaled)
        mu_orig = scaler_y_low.inverse_transform(mu.reshape(-1, 1)).ravel()
        return mu_orig[0]

    m = 2 * dim
    min_points, max_points = pso_search_extrema(
        model_func=low_fidelity_func,
        bounds=bounds,
        n_searches=m,
        pop_size=5 * dim,
        iters=30,
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
        n_candidates = max(n_needed * 100, 1000)
        candidate_pool = latin_hypercube_sampling(bounds, n_candidates)

        for _ in range(n_needed):
            if len(initial_targets) == 0:
                best_cand = candidate_pool[0]
            else:
                max_min_dist = -1.0
                best_cand = None
                np.random.shuffle(candidate_pool)
                for cand in candidate_pool[:500]:
                    dists = np.linalg.norm(initial_targets - cand, axis=1)
                    min_d = np.min(dists)
                    if min_d > max_min_dist:
                        max_min_dist = min_d
                        best_cand = cand
            initial_targets = np.vstack([initial_targets, best_cand])

    snapped_initial_points = [
        point for point in patched_snap_targets_to_low_fidelity(initial_targets, X_low) if point is not None
    ]
    initial_points = patched_fill_with_low_fidelity_points(snapped_initial_points, X_low, N)
    return initial_points


def write_high_csv(initial_points):
    rows = []
    for point in initial_points:
        key = alpha_key(point)
        if key not in corrected_map:
            raise KeyError(f'missing high-fidelity value for Alpha={key}')
        rows.append({'Alpha': key, 'CL': corrected_map[key]})
    pd.DataFrame(rows).to_csv(HIGH_CSV, index=False)


def auto_get_input(points, dim, is_initial=False, X_h=None, y_h=None):
    xs = []
    ys = []
    for point in points:
        if point is None:
            continue
        key = alpha_key(point)
        if key not in corrected_map:
            raise KeyError(f'missing high-fidelity value for Alpha={key}')
        xs.append(np.asarray(point, dtype=float))
        ys.append(corrected_map[key])
        print(f'[auto] using Corrected.csv value at Alpha={key}: CL={corrected_map[key]}', flush=True)
    return sequential.to_2d_array(xs, dim), np.asarray(ys, dtype=float)


def safe_visualize_results(X_low, y_low, X_h, y_h, fusion_model,
                           scaler_X_low, scaler_y_low,
                           bounds, dim, iterations, Fes,
                           X_val=None, y_val=None,
                           input_labels=None, output_label='Output'):
    if dim != 1 or fusion_model is None:
        return

    x_min, x_max = bounds[0]
    x_plot = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    y_plot = fusion_model.predict(x_plot, scaler_X_low=scaler_X_low, scaler_y_low=scaler_y_low)
    all_y = [*y_plot.tolist(), *y_low.tolist(), *y_h.tolist()]
    if X_val is not None and y_val is not None:
        all_y.extend(np.asarray(y_val, dtype=float).tolist())

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
        pts = ' '.join(f'{sx(x)},{sy(y)}' for x, y in points)
        return f'<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"{stroke_width}\" opacity=\"{opacity}\" points=\"{pts}\" />'

    def circles(points, color, radius):
        return '\\n'.join(
            f'<circle cx=\"{sx(x)}\" cy=\"{sy(y)}\" r=\"{radius}\" fill=\"{color}\" opacity=\"0.85\" />'
            for x, y in points
        )

    fusion_points = list(zip(x_plot[:, 0], y_plot))
    low_points = list(zip(X_low[:, 0], y_low))
    high_points = list(zip(X_h[:, 0], y_h))
    val_points = list(zip(X_val[:, 0], y_val)) if X_val is not None and y_val is not None else []
    val_points_sorted = sorted(val_points, key=lambda p: p[0])

    xlabel = input_labels[0] if input_labels else 'x'
    ylabel = output_label if output_label else 'y'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="24">Fusion result (Fes={Fes}, iterations={iterations})</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black" stroke-width="2" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black" stroke-width="2" />
<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="18">{xlabel}</text>
<text x="25" y="{height/2}" transform="rotate(-90 25,{height/2})" text-anchor="middle" font-family="Arial" font-size="18">{ylabel}</text>
{polyline(fusion_points, '#228b22', 3)}
{polyline(val_points_sorted, '#111111', 2, 0.9) if val_points_sorted else ''}
{circles(low_points, '#4169e1', 4)}
{circles(high_points, '#dc143c', 5)}
<rect x="{width-320}" y="80" width="240" height="110" fill="white" stroke="#999" />
<line x1="{width-300}" y1="110" x2="{width-250}" y2="110" stroke="#228b22" stroke-width="3" />
<text x="{width-240}" y="116" font-family="Arial" font-size="16">Fusion model</text>
<line x1="{width-300}" y1="140" x2="{width-250}" y2="140" stroke="#111111" stroke-width="2" />
<text x="{width-240}" y="146" font-family="Arial" font-size="16">Validation truth</text>
<circle cx="{width-275}" cy="168" r="4" fill="#4169e1" />
<text x="{width-240}" y="174" font-family="Arial" font-size="16">Low-fidelity</text>
<circle cx="{width-275}" cy="194" r="5" fill="#dc143c" />
<text x="{width-240}" y="200" font-family="Arial" font-size="16">High-fidelity used</text>
</svg>'''

    out_path = ROOT / f'fusion_visualization_1d_Fes{Fes}.svg'
    out_path.write_text(svg, encoding='utf-8')
    print(f'[auto] saved visualization to {out_path}', flush=True)


def run_once(Fes, final_run=False):
    sequential.snap_targets_to_low_fidelity = patched_snap_targets_to_low_fidelity
    sequential.fill_with_low_fidelity_points = patched_fill_with_low_fidelity_points
    sequential.get_input = auto_get_input
    sequential.visualize_results = safe_visualize_results if final_run else (lambda *args, **kwargs: None)

    initial_points = compute_initial_points(Fes=Fes, random_state=42, krg_corr='squar_exp')
    write_high_csv(initial_points)

    if final_run:
        answers = iter([
            str(INITIAL_CSV),
            str(Fes),
            str(CORRECTED_CSV),
            str(HIGH_CSV),
        ])

        def fake_input(prompt=''):
            try:
                answer = next(answers)
            except StopIteration:
                raise RuntimeError(f'Unexpected prompt after predefined answers: {prompt}')
            print(f'[auto-input] {prompt}{answer}', flush=True)
            return answer

        builtins.input = fake_input
        main.main()
        return None

    answers = iter([str(HIGH_CSV)])

    def fake_input(prompt=''):
        try:
            answer = next(answers)
        except StopIteration:
            raise RuntimeError(f'Unexpected prompt after predefined answers: {prompt}')
        print(f'[auto-input] {prompt}{answer}', flush=True)
        return answer

    builtins.input = fake_input
    fusion, X_h, y_h, log_df = sequential.sequential_fusion(
        cfd_csv=str(INITIAL_CSV),
        Fes=Fes,
        max_iters=50,
        random_state=42,
        krg_corr='squar_exp',
        validation_csv=str(CORRECTED_CSV),
    )

    X_val = corrected_df[['Alpha']].values.astype(float)
    y_val = corrected_df['CL'].values.astype(float)
    y_pred = fusion.predict(X_val, scaler_X_low=StandardScaler().fit(pd.read_csv(INITIAL_CSV)[['Alpha']].values.astype(float)), scaler_y_low=StandardScaler().fit(pd.read_csv(INITIAL_CSV)[['CL']].values.astype(float)))
    rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_val)))
    return {
        'Fes': Fes,
        'rmse': rmse,
        'mae': mae,
        'n_high': int(len(X_h)),
        'iterations': int(log_df['iteration'].max()) if not log_df.empty else 0,
        'initial_points': initial_points.ravel().tolist(),
    }


results = []
for Fes in [7, 9, 11, 13]:
    print(f'===== TUNE Fes={Fes} =====', flush=True)
    result = run_once(Fes=Fes, final_run=False)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    results.append(result)

best = min(results, key=lambda x: x['rmse'])
print('BEST', json.dumps(best, ensure_ascii=False), flush=True)
run_once(Fes=best['Fes'], final_run=True)

builtins.input = orig_input
sequential.snap_targets_to_low_fidelity = orig_snap
sequential.fill_with_low_fidelity_points = orig_fill
sequential.get_input = orig_get_input
sequential.visualize_results = orig_visualize
