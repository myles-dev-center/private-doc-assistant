"""Export extracted records and query results to Excel."""

from __future__ import annotations

import io
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from .structured import Schema


def build_workbook(columns: list[str], rows: list[list], title: str = "Results") -> Workbook:
    wb = Workbook()
    ws = wb.active
    ws.title = title[:31]
    ws.append(columns)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2F4F6F")
    for r in rows:
        ws.append(r)
    for i, col in enumerate(columns, 1):
        width = max([len(str(col))] + [len(str(r[i - 1])) for r in rows if r[i - 1] is not None])
        ws.column_dimensions[get_column_letter(i)].width = min(width + 2, 60)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    return wb


def xlsx_bytes(columns: list[str], rows: list[list], title: str = "Results") -> bytes:
    buf = io.BytesIO()
    build_workbook(columns, rows, title).save(buf)
    return buf.getvalue()


def export_xlsx(records: list[dict], schema: Schema, path: Path) -> None:
    cols = schema.columns
    build_workbook(cols, [[r.get(c) for c in cols] for r in records], schema.name + "s").save(path)
