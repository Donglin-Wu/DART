from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = {
    "processed": ROOT / "data" / "processed" / "wind_tunnel" / "corrected",
    "raw": ROOT / "data" / "raw" / "wind_tunnel" / "initial",
}
SURFACE_FILES = ("Ma0.3.csv", "Ma0.4.csv", "Ma0.5.csv")


def read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.reader(file))


def write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def safe_folder_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name.strip())


def build_folder_names(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    folder_names: list[str] = []
    for header in headers[1:]:
        counts[header] = counts.get(header, 0) + 1
        index = counts[header]
        suffix = f"_{index}" if headers[1:].count(header) > 1 else ""
        folder_names.append(safe_folder_name(f"{header}{suffix}"))
    return folder_names


def fix_single_point_jumps(rows: list[list[str]]) -> tuple[list[list[str]], list[tuple[float, str, float, float]]]:
    fixed_rows = [row[:] for row in rows]
    data = fixed_rows[1:]
    if len(data) < 3:
        return fixed_rows, []

    flagged_rows: set[int] = set()
    numeric_columns = len(rows[0]) - 1
    for row_index in range(1, len(data) - 1):
        candidate_count = 0
        for column_index in range(1, numeric_columns + 1):
            previous_value = float(data[row_index - 1][column_index])
            current_value = float(data[row_index][column_index])
            next_value = float(data[row_index + 1][column_index])
            interpolated_value = (previous_value + next_value) / 2
            residual = abs(current_value - interpolated_value)
            neighbor_gap = abs(next_value - previous_value)
            ratio = residual / (neighbor_gap + 1e-12)
            if residual >= 0.1 and neighbor_gap <= 0.08 and ratio >= 3:
                candidate_count += 1

        if candidate_count >= 3:
            flagged_rows.add(row_index)

    changes: list[tuple[float, str, float, float]] = []
    for row_index in sorted(flagged_rows):
        alpha = float(data[row_index][0])
        for column_index in range(1, numeric_columns + 1):
            previous_value = float(data[row_index - 1][column_index])
            current_value = float(data[row_index][column_index])
            next_value = float(data[row_index + 1][column_index])
            interpolated_value = (previous_value + next_value) / 2
            data[row_index][column_index] = format(interpolated_value, ".15g")
            header = rows[0][column_index]
            changes.append((alpha, header, current_value, interpolated_value))

    return fixed_rows, changes


def export_split_files(source_dir: Path, file_name: str, rows: list[list[str]], folder_names: list[str]) -> None:
    headers = rows[0]
    first_header = headers[0]
    data = rows[1:]
    for column_index, folder_name in enumerate(folder_names, start=1):
        output_rows: list[list[object]] = [[first_header, headers[column_index]]]
        for row in data:
            output_rows.append([row[0], row[column_index]])
        write_csv(source_dir / folder_name / file_name, output_rows)


def export_surface_files(source_dir: Path, grouped_rows: dict[str, list[list[str]]], folder_names: list[str]) -> None:
    base_rows = grouped_rows[SURFACE_FILES[0]]
    first_header = base_rows[0][0]
    headers = base_rows[0]

    for column_index, folder_name in enumerate(folder_names, start=1):
        output_rows: list[list[object]] = [["Ma", first_header, headers[column_index]]]
        for file_name in SURFACE_FILES:
            ma_value = file_name.removesuffix(".csv").replace("Ma", "")
            rows = grouped_rows[file_name]
            for row in rows[1:]:
                output_rows.append([ma_value, row[0], row[column_index]])
        write_csv(source_dir / folder_name / "Ma0.3-0.5_2D.csv", output_rows)


def main() -> None:
    for source_name, source_dir in SOURCE_DIRS.items():
        csv_files = sorted(source_dir.glob("Ma*.csv"))
        if not csv_files:
            continue

        first_rows = read_csv(csv_files[0])
        folder_names = build_folder_names(first_rows[0])
        surface_rows: dict[str, list[list[str]]] = {}
        all_changes: dict[str, list[tuple[float, str, float, float]]] = {}

        for csv_file in csv_files:
            rows = read_csv(csv_file)
            processed_rows = rows
            if source_name == "processed":
                processed_rows, changes = fix_single_point_jumps(rows)
                if changes:
                    all_changes[csv_file.name] = changes

            export_split_files(source_dir, csv_file.name, processed_rows, folder_names)
            if csv_file.name in SURFACE_FILES:
                surface_rows[csv_file.name] = processed_rows

        if all(name in surface_rows for name in SURFACE_FILES):
            export_surface_files(source_dir, surface_rows, folder_names)

        if all_changes:
            print(source_name)
            for file_name, changes in all_changes.items():
                print(file_name)
                for alpha, header, old_value, new_value in changes:
                    print(f"  alpha={alpha:g}, {header}: {old_value:.9g} -> {new_value:.9g}")


if __name__ == "__main__":
    main()
