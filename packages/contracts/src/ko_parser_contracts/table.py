"""표: 구조화된 셀 목록이 정본이고 HTML·마크다운은 파생이다. 행·열은 0부터."""

import html
from typing import Literal, Self

from pydantic import Field, model_validator

from .base import ContractModel

HeaderRole = Literal["none", "column", "row"]  # column: 위쪽 열 머리, row: 왼쪽 행 머리
CellTextSource = Literal["native", "text_layer", "ocr", "vlm"]


class Cell(ContractModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    text: str = ""
    header: HeaderRole = "none"
    text_source: CellTextSource


class Table(ContractModel):
    n_rows: int = Field(ge=1)
    n_cols: int = Field(ge=1)
    cells: tuple[Cell, ...]

    @model_validator(mode="after")
    def _check_coverage(self) -> Self:
        owner: set[tuple[int, int]] = set()
        anchors: set[tuple[int, int]] = set()
        for cell in self.cells:
            if (cell.row, cell.col) in anchors:
                raise ValueError(f"duplicate cell anchor at ({cell.row}, {cell.col})")
            anchors.add((cell.row, cell.col))
            if cell.row + cell.rowspan > self.n_rows or cell.col + cell.colspan > self.n_cols:
                raise ValueError(f"cell at ({cell.row}, {cell.col}) exceeds table bounds")
            for r in range(cell.row, cell.row + cell.rowspan):
                for k in range(cell.col, cell.col + cell.colspan):
                    if (r, k) in owner:
                        raise ValueError(f"cells overlap at ({r}, {k})")
                    owner.add((r, k))
        missing = [(r, k) for r in range(self.n_rows) for k in range(self.n_cols) if (r, k) not in owner]
        if missing:
            raise ValueError(f"uncovered grid positions: {missing[:5]}")
        return self

    def _row_anchors(self, row: int) -> list[Cell]:
        return sorted((cell for cell in self.cells if cell.row == row), key=lambda cell: cell.col)

    def to_grid(self) -> tuple[tuple[str, ...], ...]:
        """병합 영역의 모든 칸에 앵커 셀 글자를 반복한다."""
        grid = [[""] * self.n_cols for _ in range(self.n_rows)]
        for cell in self.cells:
            for r in range(cell.row, cell.row + cell.rowspan):
                for k in range(cell.col, cell.col + cell.colspan):
                    grid[r][k] = cell.text
        return tuple(tuple(row) for row in grid)

    def plain_text(self) -> str:
        """행마다 그 행에서 시작하는 셀 글자를 탭으로 잇고 행은 줄바꿈으로 잇는다. 병합 셀 글자는 한 번만."""
        return "\n".join("\t".join(cell.text for cell in self._row_anchors(r)) for r in range(self.n_rows))

    def _header_row_count(self) -> int:
        count = 0
        for r in range(self.n_rows):
            anchors = self._row_anchors(r)
            if anchors and all(cell.header == "column" for cell in anchors):
                count += 1
            else:
                break
        return count

    def _row_html(self, row: int) -> str:
        parts = ["<tr>"]
        for cell in self._row_anchors(row):
            tag = "th" if cell.header != "none" else "td"
            attrs = ""
            if cell.rowspan > 1:
                attrs += f' rowspan="{cell.rowspan}"'
            if cell.colspan > 1:
                attrs += f' colspan="{cell.colspan}"'
            text = html.escape(cell.text, quote=True).replace("\n", "<br>")
            parts.append(f"<{tag}{attrs}>{text}</{tag}>")
        parts.append("</tr>")
        return "".join(parts)

    def to_html(self) -> str:
        """허용 태그 table/thead/tbody/tr/th/td, 속성 rowspan/colspan만. 위쪽의 열 머리 행은 thead."""
        head = self._header_row_count()
        out = ["<table>"]
        if head:
            out += ["<thead>", *(self._row_html(r) for r in range(head)), "</thead>"]
        out += ["<tbody>", *(self._row_html(r) for r in range(head, self.n_rows)), "</tbody>", "</table>"]
        return "".join(out)

    def to_markdown(self) -> str:
        """손실 변환: 병합 셀 글자를 덮인 칸마다 반복하고 첫 행을 머리행으로 쓴다."""

        def esc(text: str) -> str:
            return text.replace("|", "\\|").replace("\n", "<br>")

        grid = self.to_grid()
        lines = [
            "| " + " | ".join(esc(x) for x in grid[0]) + " |",
            "| " + " | ".join("---" for _ in range(self.n_cols)) + " |",
        ]
        lines += ["| " + " | ".join(esc(x) for x in row) + " |" for row in grid[1:]]
        return "\n".join(lines)
