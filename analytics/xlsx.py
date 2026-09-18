"""Write the question/thread/summary rows to an .xlsx workbook.

openpyxl is NOT installed as a project dependency; it's imported lazily inside
write_workbook, so run xlsx output with `uv run --with openpyxl` (or pip install it).
"""

import math
import re

# openpyxl has no real autofit (its `bestFit` flag does not measure or resize), so
# width is estimated from character counts. Width ≈ chars of the default font.
_MAX_COL_WIDTH = 70  # cap so a long free-text cell doesn't blow the column out
_COL_PADDING = 2  # a little breathing room beyond the widest data value
_HEADER_BOLD_FACTOR = 1.2  # the header row is bold, so it renders wider than plain
_FILTER_ARROW_WIDTH = 3  # room for the auto-filter dropdown button over the header

# openpyxl turns a string starting with "=" into a formula (which then shows as an
# error like Err:510); these leading chars also let a spreadsheet treat a cell as a
# formula. Any such string cell is forced to plain text instead.
_FORMULA_PREFIXES = ("=", "+", "-", "@")

# openpyxl rejects ASCII control chars and caps a cell at 32,767 chars. Both are
# xlsx limits, so text is cleaned here rather than at extraction — the BigQuery
# tables keep whatever the database holds.
_ILLEGAL_XLSX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_CELL_LIMIT = 32_767
_TRUNCATED_SUFFIX = "…[truncated]"


def _cell(value):
    """Make a value safe for a cell: booleans as 1/0, no control chars, capped length."""
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, str):
        return value
    value = _ILLEGAL_XLSX.sub("", value)
    if len(value) > _CELL_LIMIT:
        value = value[: _CELL_LIMIT - len(_TRUNCATED_SUFFIX)] + _TRUNCATED_SUFFIX
    return value


def _append_row(sheet, values) -> None:
    """Append a row, forcing any formula-looking string cell to plain text.

    Without this, a divider like "===== ... =====" (or user text starting with `=`)
    is stored as a formula and renders as an error when the file is opened.
    """
    sheet.append([_cell(v) for v in values])
    for cell in sheet[sheet.max_row]:
        if isinstance(cell.value, str) and cell.value.startswith(_FORMULA_PREFIXES):
            cell.data_type = "s"


def _fit_width(header, values) -> float:
    """Column width fitting the widest data cell and the bold header (plus filter arrow)."""
    longest = max((len(str(v)) for v in values if v is not None), default=0)
    header_need = (
        math.ceil(len(str(header)) * _HEADER_BOLD_FACTOR) + _FILTER_ARROW_WIDTH
    )
    return min(max(longest + _COL_PADDING, header_need), _MAX_COL_WIDTH)


def write_workbook(path: str, question_rows, thread_rows, summary_rows) -> None:
    from openpyxl import Workbook  # type: ignore
    from openpyxl.styles import Font  # type: ignore
    from openpyxl.utils import get_column_letter  # type: ignore

    wb = Workbook()

    # Summary sheet
    ws = wb.active
    ws.title = "Resumo"
    _append_row(ws, ["Métrica", "Valor"])
    ws["A1"].font = ws["B1"].font = Font(bold=True)
    for label, value in summary_rows:
        _append_row(ws, [label, value])
    ws.column_dimensions["A"].width = _fit_width(
        "Métrica", [label for label, _ in summary_rows]
    )
    ws.column_dimensions["B"].width = _fit_width(
        "Valor", [_cell(value) for _, value in summary_rows]
    )

    def _sheet(title: str, rows: list[dict]):
        sheet = wb.create_sheet(title)
        if not rows:
            sheet.append(["(no data)"])
            return
        headers = list(rows[0].keys())
        _append_row(sheet, headers)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for row in rows:
            _append_row(sheet, [row[h] for h in headers])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(rows) + 1}"
        for idx, header in enumerate(headers, start=1):
            values = [_cell(row[header]) for row in rows]
            sheet.column_dimensions[get_column_letter(idx)].width = _fit_width(
                header, values
            )

    _sheet("Perguntas", question_rows)
    _sheet("Conversas", thread_rows)

    wb.save(path)
