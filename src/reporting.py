"""Portable Markdown-to-DOCX renderer for generated model evaluation reports."""
from pathlib import Path

from docx import Document
from docx.shared import Pt


def _split_table_row(line):
    """Split a Markdown pipe row while tolerating optional outer pipes."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def markdown_to_docx(markdown_path, output_path):
    """Render the project's simple Markdown report into a readable Word file."""
    markdown_path = Path(markdown_path)
    output_path = Path(output_path)
    document = Document()
    document.core_properties.title = "Building Energy Model Evaluation"
    document.core_properties.subject = "Chronological and building-group model validation"

    table_lines = []

    def flush_table():
        nonlocal table_lines
        if not table_lines:
            return
        rows = [_split_table_row(line) for line in table_lines]
        rows = [row for row in rows if not all(set(cell) <= set("-: ") for cell in row)]
        if rows:
            width = max(map(len, rows))
            table = document.add_table(rows=0, cols=width)
            table.style = "Light Shading Accent 1"
            for values in rows:
                cells = table.add_row().cells
                for index, value in enumerate(values):
                    cells[index].text = value
        table_lines = []

    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            document.add_heading(line[2:], level=0)
        elif line.startswith("## "):
            document.add_heading(line[3:], level=1)
        elif line.startswith("- "):
            document.add_paragraph(line[2:], style="List Bullet")
        elif line.startswith("> "):
            document.add_paragraph(line[2:], style="Quote")
        elif line.strip():
            document.add_paragraph(line)
    flush_table()

    for paragraph in document.paragraphs:
        for run in paragraph.runs:
            run.font.size = Pt(10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)
    return output_path


def building_error_summary(predictions):
    """Aggregate actual-vs-predicted diagnostics per building.

    Required columns: building_id, actual, predicted. Zero actual readings are
    excluded from APE; WAPE is undefined when a building's absolute actual sum
    is zero. Residual is actual minus prediction (positive = underprediction).
    """
    import numpy as np
    import pandas as pd

    required = {"building_id", "actual", "predicted"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError("Building error analysis missing columns: " + ", ".join(sorted(missing)))
    frame = predictions[list(required)].copy()
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    frame["predicted"] = pd.to_numeric(frame["predicted"], errors="coerce")
    frame = frame.dropna(subset=["building_id", "actual", "predicted"])
    frame = frame[np.isfinite(frame["actual"]) & np.isfinite(frame["predicted"])].copy()
    if frame.empty:
        return pd.DataFrame(columns=[
            "building_id", "test_rows", "mae", "rmse", "mean_error", "wape",
            "median_absolute_percentage_error", "ape_valid_rows", "bias_direction"
        ])
    frame["residual"] = frame["actual"] - frame["predicted"]
    frame["absolute_error"] = frame["residual"].abs()
    denominator = frame["actual"].abs()
    frame["ape"] = np.where(denominator > 0, frame["absolute_error"] / denominator, np.nan)
    summary = frame.groupby("building_id").agg(
        test_rows=("residual", "size"),
        mae=("absolute_error", "mean"),
        rmse=("residual", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        mean_error=("residual", "mean"),
        absolute_error_sum=("absolute_error", "sum"),
        actual_abs_sum=("actual", lambda values: float(values.abs().sum())),
        median_absolute_percentage_error=("ape", "median"),
        ape_valid_rows=("ape", "count"),
    ).reset_index()
    summary["wape"] = np.where(
        summary["actual_abs_sum"] > 0,
        summary["absolute_error_sum"] / summary["actual_abs_sum"],
        np.nan,
    )
    summary["bias_direction"] = np.where(
        summary["mean_error"] > 0, "underprediction",
        np.where(summary["mean_error"] < 0, "overprediction", "balanced"),
    )
    return summary.drop(columns=["absolute_error_sum", "actual_abs_sum"]).sort_values(
        "mae", ascending=False
    ).reset_index(drop=True)
