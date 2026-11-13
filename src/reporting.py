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
