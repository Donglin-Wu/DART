# DART

DART is a Python implementation of a dual-fidelity sequential fusion workflow for combining low-fidelity simulation data and high-fidelity experiment data. The repository includes the core DART package, interactive and command-line entry points, and experiment scripts used for convergence and ablation verification.

## Repository Structure

- `DART/` contains the reusable fusion workflow, Kriging surrogate model, sequential sampling strategy, visualization utilities, command-line entry point, and Tkinter GUI.
- `ConvergenceVerification/` contains scripts for convergence verification experiments.
- `AblationStudy/` contains scripts for ablation experiments.
- `FunctionExperiment/` contains one-dimensional function experiment resources.
- `CFDVerification/` contains CFD verification resources.
- `WindTunnelData/` contains wind-tunnel related resources.

## Environment

Use Python 3.9 or later. Create a virtual environment before installing dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Command-line Usage

Run the main workflow from the repository root:

```bash
python -m DART.main
```

The program prompts for:

- The low-fidelity CSV file path.
- The high-fidelity budget limit `BHF`.
- An optional full-space high-fidelity validation CSV file path.
- Initial and sequential high-fidelity values when requested.

## GUI Usage

Run the interactive GUI from the repository root:

```bash
python -m DART.fusion_gui
```

Use the GUI to select input CSV files, set the high-fidelity budget, submit initial high-fidelity data, and enter recommended high-fidelity values during sequential iterations.

## Standalone Prediction

After a workflow run produces `trained_model.pkl`, generate predictions with:

```bash
python -m DART.predict_and_plot_from_pkl --model path\to\trained_model.pkl --input path\to\input.csv
```

Optional arguments:

- `--truth`: high-fidelity truth CSV for regression-fit scatter plotting.
- `--output-dir`: directory for generated outputs.
- `--output-csv`: custom prediction CSV path.
- `--scatter-svg`: custom regression-fit scatter SVG path.

## Input Data Format

CSV files should place input variables in the first columns and the output variable in the last column. The code treats the final CSV column as the target output by default.

## Outputs

Workflow runs create timestamped result folders containing logs, selected high-fidelity samples, model parameters, trained model files, prediction CSV files, and SVG visualizations.

Generated caches and temporary outputs are excluded by `.gitignore`.

