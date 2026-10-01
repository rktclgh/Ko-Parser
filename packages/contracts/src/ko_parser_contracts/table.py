"""표: 구조화된 셀 목록이 정본이고 HTML·마크다운은 파생이다. 행·열은 0부터."""

import html
from itertools import islice
from typing import Literal, Self

from pydantic import Field, model_validator

from .base import ContractModel

HeaderRole = Literal["none", "column", "row"]  # column: 위쪽 열 머리, row: 왼쪽 행 머리
CellTextSource = Literal["native", "text_layer", "ocr", "vlm"]
MAX_TABLE_CELLS = 100_000  # 그리드 칸 수 n_rows×n_cols 상한; 넘는 표는 엔진이 블록을 나눈다
MAX_TABLE_EXPANDED_CHARS = 10_000_000  # 병합 셀 글자를 덮인 칸마다 펼친 총 글자 수 상한; to_grid·to_markdown 출력 크기를 묶는다


def _lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


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
    cells: tuple[Cell, ...] = Field(max_length=MAX_TABLE_CELLS)

    @model_validator(mode="after")
    def _check_coverage(self) -> Self:
        if self.n_rows * self.n_cols > MAX_TABLE_CELLS:
            raise ValueError(f"table grid {self.n_rows}x{self.n_cols} exceeds {MAX_TABLE_CELLS} cells")
        owner: set[tuple[int, int]] = set()
        anchors: set[tuple[int, int]] = set()
        expanded = 0
        for cell in self.cells:
            if (cell.row, cell.col) in anchors:
                raise ValueError(f"duplicate cell anchor at ({cell.row}, {cell.col})")
            anchors.add((cell.row, cell.col))
            if cell.row + cell.rowspan > self.n_rows or cell.col + cell.colspan > self.n_cols:
                raise ValueError(f"cell at ({cell.row}, {cell.col}) exceeds table bounds")
            expanded += len(cell.text) * cell.rowspan * cell.colspan
            if expanded > MAX_TABLE_EXPANDED_CHARS:
                raise ValueError(f"table expanded text exceeds {MAX_TABLE_EXPANDED_CHARS} chars")
            for r in range(cell.row, cell.row + cell.rowspan):
                for k in range(cell.col, cell.col + cell.colspan):
                    if (r, k) in owner:
                        raise ValueError(f"cells overlap at ({r}, {k})")
                    owner.add((r, k))
        if len(owner) != self.n_rows * self.n_cols:
            uncovered = ((r, k) for r in range(self.n_rows) for k in range(self.n_cols) if (r, k) not in owner)
            raise ValueError(f"uncovered grid positions: {list(islice(uncovered, 5))}")
        return self

    def _rows(self) -> list[list[Cell]]:
        """행마다 그 행에서 시작하는 셀을 열 순서로 한 번에 묶는다."""
        rows: list[list[Cell]] = [[] for _ in range(self.n_rows)]
        for cell in self.cells:
            rows[cell.row].append(cell)
        for row in rows:
            row.sort(key=lambda cell: cell.col)
        return rows

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
        return "\n".join("\t".join(cell.text for cell in row) for row in self._rows())

    def _header_row_count(self, rows: list[list[Cell]]) -> int:
        count = 0
        for anchors in rows:
            if anchors and all(cell.header == "column" for cell in anchors):
                count += 1
            else:
                break
        blocked = {b for cell in self.cells for b in range(cell.row + 1, cell.row + cell.rowspan)}
        while count in blocked:  # rowspan은 행 그룹 경계를 넘을 수 없다
            count -= 1
        return count

    @staticmethod
    def _row_html(row: list[Cell]) -> str:
        parts = ["<tr>"]
        for cell in row:
            tag = "th" if cell.header != "none" else "td"
            attrs = ""
            if cell.rowspan > 1:
                attrs += f' rowspan="{cell.rowspan}"'
            if cell.colspan > 1:
                attrs += f' colspan="{cell.colspan}"'
            text = html.escape(_lf(cell.text), quote=True).replace("\n", "<br>")
            parts.append(f"<{tag}{attrs}>{text}</{tag}>")
        parts.append("</tr>")
        return "".join(parts)

    def to_html(self) -> str:
        """허용 태그 table/thead/tbody/tr/th/td, 속성 rowspan/colspan만. 위쪽의 열 머리 행은 thead."""
        rows = self._rows()
        head = self._header_row_count(rows)
        out = ["<table>"]
        if head:
            out += ["<thead>", *(self._row_html(row) for row in rows[:head]), "</thead>"]
        out += ["<tbody>", *(self._row_html(row) for row in rows[head:]), "</tbody>", "</table>"]
        return "".join(out)

    def to_markdown(self) -> str:
        """손실 변환: 병합 셀 글자를 덮인 칸마다 반복하고 첫 행을 머리행으로 쓴다.

        출력은 신뢰할 수 없는 텍스트다(브라우저에 렌더할 땐 raw HTML을 끄거나 정화할 것).
        """

        def esc(text: str) -> str:
            return _lf(text).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")

        grid = self.to_grid()
        lines = [
            "| " + " | ".join(esc(x) for x in grid[0]) + " |",
            "| " + " | ".join("---" for _ in range(self.n_cols)) + " |",
        ]
        lines += ["| " + " | ".join(esc(x) for x in row) + " |" for row in grid[1:]]
        return "\n".join(lines)
