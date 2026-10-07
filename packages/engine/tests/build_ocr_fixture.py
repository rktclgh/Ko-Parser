"""OCR 예제(스캔 흉내 PDF) 생성기. 한 번 만들어 커밋한다(바이트 --check 대상이 아니다).

    uv run python packages/engine/tests/build_ocr_fixture.py

reportlab으로 그린 쪽(HYGothic, 미임베드)을 PDFium으로 200 DPI 회색 그림으로 그리고, 그 그림을 새 PDF 한 쪽에
가득 넣은 뒤 쪽 번호만 보이는 글자로 덧쓴다(스캔 PDF에 쪽 번호를 덧쓴 모양). 미임베드 글꼴은 OS 글꼴로 그려지므로
그림 바이트가 OS마다 다르다. 그래서 만든 PDF를 고정 입력으로 커밋하고 test_pdf_ocr_fixture.py가 그 파일을 읽는다
(그림이 고정이라 OS마다 다른 것은 onnxruntime 부동소수 차이뿐이다)."""

import io
from pathlib import Path

from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from hanji.formats.pdf.scan import render

OUT = Path(__file__).resolve().parent / "fixtures" / "ocr" / "scanned.pdf"
FONT = "HYGothic-Medium"
A5 = (420.0, 595.0)
LINES = [(50, 540, 18, "스캔 문서 읽기 시험"), (50, 500, 12, "이 쪽은 그림 한 장으로 된 스캔 쪽이다."),
         (50, 482, 12, "그림 속 글자는 문단 블록이 된다."), (50, 440, 12, "두 번째 문단은 한 줄이다.")]
PAGE_NUMBER = (205, 30, 9, "1")

pdfmetrics.registerFont(UnicodeCIDFont(FONT))


def canvas_pdf(draw) -> bytes:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A5, invariant=1, pageCompression=0)
    draw(c)
    c.showPage()
    c.save()
    return buf.getvalue()


def source(c: Canvas) -> None:
    for x, y, size, s in [*LINES, PAGE_NUMBER]:
        c.setFont(FONT, size)
        c.drawString(x, y, s)


def build() -> bytes:
    image = render(canvas_pdf(source), "source.pdf", 0).convert("L")

    def scanned(c: Canvas) -> None:
        c.drawImage(ImageReader(image), 0, 0, width=A5[0], height=A5[1])
        x, y, size, s = PAGE_NUMBER
        c.setFont(FONT, size)
        c.drawString(x, y, s)

    return canvas_pdf(scanned)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(build())
    print(OUT)
