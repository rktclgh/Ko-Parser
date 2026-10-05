"""PDF 골든 예제 생성기(reportlab, 합성 한국어 샘플만).

    uv run python packages/engine/tests/build_pdf_fixtures.py          # fixtures/pdf/ 재생성
    uv run python packages/engine/tests/build_pdf_fixtures.py --check  # 최신인지 확인(바이트 비교)

글꼴은 reportlab 내장 한국어 CID 글꼴(HYGothic-Medium, 미임베드)이라 저장소에 글꼴 파일이 없다.
invariant 모드와 압축 없음(zlib 판 차이 회피)으로 바이트가 결정적이다. 그림은 아래 JPEG를 그대로 넣는다(DCTDecode).
"""

import argparse
import hashlib
import io
import json
import sys
from collections.abc import Callable
from pathlib import Path

from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from ko_parser.core import build_tree
from ko_parser.formats.pdf import PdfParser
from ko_parser_contracts import DocumentTree, SourceInfo

ROOT = Path(__file__).resolve().parent / "fixtures" / "pdf"
FONT = "HYGothic-Medium"
A4 = (595.0, 842.0)
FILL, FILL_STROKE, INVISIBLE = 0, 2, 3  # 렌더 모드: 보통, 채우기+윤곽(굵게 흉내), 숨김
# 8×8 회색 JPEG(Pillow로 한 번 만든 바이트를 고정)
GRAY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300100b0c0e0c0a100e0d0e1211101318281a181616183123251d283a333d"
    "3c3933383740485c4e404457453738506d51575f626768673e4d71797064785c656763ffc0000b080008000801011100ffc40014000100"
    "000000000000000000000000000005ffc40014100100000000000000000000000000000000ffda0008010100003f0041ffd9")

pdfmetrics.registerFont(UnicodeCIDFont(FONT))


def text(c: Canvas, x: float, y: float, size: float, s: str, mode: int = FILL) -> None:
    """렌더 모드(Tr)는 그래픽 상태라 BT/ET 밖으로 이어진다. reportlab은 0 Tr을 생략하므로 q/Q로 감싼다."""
    c.saveState()
    t = c.beginText(x, y)
    t.setFont(FONT, size)
    t.setTextRenderMode(mode)
    t.textOut(s)
    c.drawText(t)
    c.restoreState()


def report(c: Canvas) -> None:
    """제목 4단계(마지막은 굵게), □/○/- 목록, 내어쓰기 이어짐 줄, 여러 줄 문단, 큰 글자 3줄 문단, 줄 조각."""
    text(c, 72, 770, 20, "2026년 사업 계획")
    text(c, 72, 745, 20, "(초안)")
    text(c, 72, 700, 16, "1. 추진 배경")
    text(c, 72, 670, 11, "본 계획은 2026년 지역 디지털 전환 사업의 추진 방향과")
    text(c, 72, 654, 11, "세부 일정을 정리한 것으로, 참여 기관 모집부터 성과 점검까지")
    text(c, 72, 638, 11, "단계별 과제를 담는다.")
    text(c, 72, 600, 13, "가. 세부 목표")
    text(c, 72, 570, 11, "□ 첫째, 참여 기관을 모집한다.")
    text(c, 88.5, 554, 11, "모집 공고는 1분기에 낸다.")
    text(c, 72, 530, 11, "○ 둘째, 선정 결과를 통보한다.")
    text(c, 72, 506, 11, "- 셋째, 협약을 체결한다.")
    text(c, 72, 470, 11, "본 계획의 세부 내용은 사업 여건에 따라")
    text(c, 72, 454, 11, "변경될 수 있다.")
    c.showPage()
    text(c, 72, 770, 16, "2. 예산")
    text(c, 72, 740, 12, "핵심 지표", FILL_STROKE)
    text(c, 72, 715, 11, "총사업비는 4,250,000원이다.")
    text(c, 72, 680, 16, "큰 글자로 쓴 강조 문단은")
    text(c, 72, 660, 16, "세 줄 이상 이어지면")
    text(c, 72, 640, 16, "제목이 아니라 문단이다.")
    text(c, 72, 600, 11, "구분")
    text(c, 300, 600, 11, "금액(원)")
    text(c, 72, 584, 11, "인건비")
    text(c, 300, 584, 11, "4,250,000")
    c.showPage()


def header_footer(c: Canvas) -> None:
    """3쪽: 머리말·꼬리말 반복(쪽 번호만 다름), 1쪽에만 있는 위 여백 글자는 반복이 아니다."""
    bodies = ["첫째 쪽 본문이다. 사업 개요를 설명한다.", "둘째 쪽 본문이다. 추진 일정을 설명한다.",
              "셋째 쪽 본문이다. 예산 계획을 설명한다."]
    for n, body in enumerate(bodies, 1):
        text(c, 72, 815, 9, "2026년 사업 계획 보고")
        if n == 1:
            text(c, 480, 815, 9, "대외비")
        text(c, 72, 700, 11, body)
        text(c, 282, 30, 9, f"- {n} -")
        c.showPage()


def scanned_invisible(c: Canvas) -> None:
    """숨은 글자층(렌더 모드 3)만 있고 보이는 글자는 쪽 번호뿐 → scanned. 숨은 글자는 버리고 쪽 번호는 블록."""
    for i, line in enumerate(["스캔한 쪽 위에 얹은 글자층이다.", "사람 눈에는 보이지 않는다.", "판정은 scanned다."]):
        text(c, 72, 770 - 20 * i, 11, line, INVISIBLE)
    text(c, 282, 30, 9, "- 1 -")
    c.showPage()


def image_page(c: Canvas) -> None:
    """쪽 면적의 약 30%를 덮는 그림 + 쪽 번호만 → scanned(그림 쪽 규칙), 쪽 번호는 블록. 스펙 기준값(0.5)이면
    digital이 될 쪽이라 공공누리 실측으로 낮춘 기준(0.15·50자)을 고정한다."""
    c.drawImage(ImageReader(io.BytesIO(GRAY_JPEG)), 97.5, 300, width=400, height=370)
    text(c, 282, 30, 9, "- 3 -")
    c.showPage()


def ruled_table(c: Canvas) -> None:
    """제목·문단 뒤 선 있는 표 하나(머리행 배경, 가로 병합 '상반기', 세로 병합 '사업', 칸 안 두 줄)와 뒤 문단."""
    text(c, 72, 770, 16, "1. 추진 실적")
    text(c, 72, 745, 11, "분기별 실적은 아래 표와 같다.")
    xs, ys = [72, 172, 272, 372, 472], [720, 696, 672, 648, 624]  # 칸 높이 24pt
    c.saveState()
    c.setFillGray(0.85)
    c.rect(xs[0], ys[1], xs[-1] - xs[0], ys[0] - ys[1], stroke=0, fill=1)
    c.restoreState()
    for j, y in enumerate(ys):
        c.line(xs[1] if j == 2 else xs[0], y, xs[-1], y)  # 1열 2·3행 사이는 선이 없다(세로 병합)
    for x in xs:
        c.line(x, ys[1] if x == 272 else ys[0], x, ys[-1])  # 머리행 2·3열 사이는 선이 없다(가로 병합)
    text(c, 105, 703, 11, "구분")
    text(c, 255.5, 703, 11, "상반기")
    text(c, 405, 703, 11, "비고")
    text(c, 105, 668, 11, "사업")
    for x, row in zip((205, 305), (("1분기", "3건", "3건"), ("2분기", "5건", "5건"))):
        for y, s in zip((679, 655, 631), row):
            text(c, x, y, 11, s)
    text(c, 105, 631, 11, "합계")
    text(c, 405, 637, 9, "누적")
    text(c, 405, 627, 9, "8건")
    text(c, 72, 590, 11, "실적은 분기마다 갱신한다.")
    c.showPage()


def empty(c: Canvas) -> None:
    """글자도 그림도 없는 쪽 → digital, 블록 없음."""
    c.showPage()


SAMPLES: dict[str, Callable[[Canvas], None]] = {
    "report.pdf": report, "header_footer.pdf": header_footer, "scanned_invisible.pdf": scanned_invisible,
    "image_page.pdf": image_page, "empty.pdf": empty, "table.pdf": ruled_table,
}


def render(draw: Callable[[Canvas], None]) -> bytes:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4, invariant=1, pageCompression=0)
    draw(c)
    c.save()
    return buf.getvalue()


def document_id(name: str) -> str:
    return "fx-pdf-" + Path(name).stem


def golden_tree(name: str, data: bytes) -> DocumentTree:
    """엔진 ingest와 같은 SourceInfo(page_count = 쪽 수). OCR은 끈다: OCR 결과는 CPU마다 조금씩 달라 바이트 비교 대상이
    아니다(OCR은 test_pdf_ocr.py가 허용 오차로 본다)."""
    parsed = PdfParser(ocr=False).parse(data, name)
    source = SourceInfo(name=name, mime=parsed.mime, content_hash="sha256:" + hashlib.sha256(data).hexdigest(),
                        page_count=len(parsed.pages) or None)
    return build_tree(parsed, document_id(name), 1, source)


def build_all() -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for name, draw in SAMPLES.items():
        data = render(draw)
        out[f"inputs/{name}"] = data
        out[f"expected/{Path(name).stem}.json"] = (json.dumps(golden_tree(name, data).model_dump(mode="json"),
                                                              ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ko-parser-engine PDF fixtures")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    stale = []
    for rel, data in build_all().items():
        path = ROOT / rel
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                stale.append(rel)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print(path)
    for rel in stale:
        print(f"stale fixture: {rel}", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
