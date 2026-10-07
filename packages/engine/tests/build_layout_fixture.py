"""레이아웃 예제(합성 PDF) 생성기. 한 번 만들어 커밋한다.

    uv run python packages/engine/tests/build_layout_fixture.py

reportlab(invariant, 압축 없음)으로 두 쪽을 그린다. 1쪽: 막대 차트(선·도형) + 아래 캡션, 사진(이미지 객체, Pillow로 그린
결정적 그림) + 아래 캡션, 머리 오른쪽 로고(작은 이미지 = 장식). 2쪽: 눈금 격자가 있는 꺾은선 차트(find_tables가 격자를
표로 잡는다 → 그림이 이긴다) + 위 캡션, 선 있는 표 + 표 제목(그림 캡션이 아니다). 글자는 지어낸 것이다.
PDF 바이트는 결정적이지만 미임베드 한글 글꼴은 OS 글꼴로 그려져 모델 입력(렌더)이 OS마다 조금 다르다: 테스트는 상자를
IoU로 본다(test_pdf_layout_fixture.py)."""

import io
import math
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

OUT = Path(__file__).resolve().parent / "fixtures" / "layout" / "figures.pdf"
FONT = "HYGothic-Medium"
A4 = (595.0, 842.0)

pdfmetrics.registerFont(UnicodeCIDFont(FONT))


def put(c: Canvas, x: float, y: float, size: float, s: str, font: str = FONT) -> None:
    c.setFont(font, size)
    c.drawString(x, y, s)


def photo() -> Image.Image:
    """사진 흉내: 하늘·땅 그라데이션과 해·나무(결정적)."""
    w, h = 480, 300
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            if y < h * 0.6:
                px[x, y] = (90 + y // 3, 150 + y // 4, 230)
            else:
                px[x, y] = (60 + (x * 7 + y * 3) % 40, 140 + (x * 3 + y * 5) % 50, 60)
    d = ImageDraw.Draw(img)
    d.ellipse((360, 30, 430, 100), fill=(250, 220, 60))
    for k, x in enumerate((60, 150, 260)):
        d.rectangle((x + 18, 150, x + 32, 200), fill=(110, 70, 30))
        d.ellipse((x - 10, 80 + 10 * k, x + 60, 170), fill=(30, 110 + 20 * k, 40))
    return img


def logo() -> Image.Image:
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill=(200, 30, 40))
    d.rectangle((24, 14, 40, 50), fill=(255, 255, 255))
    return img


def bar_chart(c: Canvas, x0: float, y0: float, w: float, h: float) -> None:
    c.setLineWidth(1)
    c.line(x0, y0, x0 + w, y0)
    c.line(x0, y0, x0, y0 + h)
    values = [32, 55, 41, 70, 63]
    colors = [(0.20, 0.45, 0.75), (0.85, 0.45, 0.15), (0.30, 0.65, 0.35), (0.70, 0.25, 0.30), (0.50, 0.40, 0.70)]
    bw = w / (len(values) * 1.6)
    for i, (v, col) in enumerate(zip(values, colors)):
        bx = x0 + bw * 0.6 + i * bw * 1.6
        c.setFillColorRGB(*col)
        c.rect(bx, y0, bw, h * v / 80, stroke=0, fill=1)
        c.setFillColorRGB(0, 0, 0)
        put(c, bx + bw * 0.25, y0 - 12, 8, f"{i + 1}Q", "Helvetica")
        put(c, bx + bw * 0.2, y0 + h * v / 80 + 3, 7, str(v), "Helvetica")
    for k in range(0, 81, 20):
        put(c, x0 - 16, y0 + h * k / 80 - 3, 7, str(k), "Helvetica")


def line_chart(c: Canvas, x0: float, y0: float, w: float, h: float) -> None:
    """눈금 격자(가로 5·세로 6) + 꺾은선 + 점마다 값 이름표 + 달 이름표. 격자와 이름표가 표처럼 보인다."""
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.setLineWidth(0.5)
    for k in range(1, 6):
        c.line(x0, y0 + h * k / 5, x0 + w, y0 + h * k / 5)
    for k in range(1, 7):
        c.line(x0 + w * k / 6, y0, x0 + w * k / 6, y0 + h)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(1)
    c.line(x0, y0, x0 + w, y0)
    c.line(x0, y0, x0, y0 + h)
    pts = [(x0 + w * i / 11, y0 + h * (0.3 + 0.25 * math.sin(i / 1.7) + 0.03 * i)) for i in range(12)]
    c.setStrokeColorRGB(0.15, 0.4, 0.8)
    c.setLineWidth(2)
    path = c.beginPath()
    path.moveTo(*pts[0])
    for q in pts[1:]:
        path.lineTo(*q)
    c.drawPath(path, stroke=1, fill=0)
    c.setFillColorRGB(0.15, 0.4, 0.8)
    for q in pts:
        c.circle(q[0], q[1], 2.5, stroke=0, fill=1)
    c.setFillColorRGB(0, 0, 0)
    for i, q in enumerate(pts):
        put(c, q[0] - 6, q[1] + 5, 7, str(100 + 7 * i), "Helvetica")
    c.setStrokeColorRGB(0, 0, 0)
    for i in range(0, 12, 2):
        put(c, x0 + w * i / 11 - 4, y0 - 12, 7, f"{i + 1}월")


def table(c: Canvas, x0: float, top: float, cols: list[float], rows: list[list[str]]) -> None:
    xs = [x0]
    for width in cols:
        xs.append(xs[-1] + width)
    ys = [top - 22 * i for i in range(len(rows) + 1)]
    c.setLineWidth(0.8)
    for y in ys:
        c.line(xs[0], y, xs[-1], y)
    for x in xs:
        c.line(x, ys[0], x, ys[-1])
    for r, row in enumerate(rows):
        for k, s in enumerate(row):
            put(c, xs[k] + 6, ys[r] - 15, 10, s)


def page1(c: Canvas) -> None:
    c.drawImage(ImageReader(logo()), 520, 790, width=30, height=30)
    put(c, 60, 790, 16, "합성 보고서: 그림 예제")
    put(c, 60, 760, 10.5, "이 쪽은 레이아웃 시험을 위한 지어낸 글이다. 막대 차트와 사진이 있다.")
    put(c, 60, 744, 10.5, "그림 아래에는 캡션이 붙는다. 로고는 그림이 아니다.")
    bar_chart(c, 110, 520, 380, 190)
    put(c, 200, 480, 10, "그림 1. 분기별 처리 건수(단위: 건)")
    c.drawImage(ImageReader(photo()), 150, 190, width=300, height=187.5)
    put(c, 220, 170, 10, "그림 2. 현장 사진")
    put(c, 60, 120, 10.5, "사진 아래 문단은 그대로 문단으로 남는다.")
    put(c, 290, 40, 9, "- 1 -")


def page2(c: Canvas) -> None:
    put(c, 60, 790, 14, "2. 월별 추이와 예산")
    put(c, 200, 760, 10, "그림 3. 월별 이용자 추이")
    line_chart(c, 100, 560, 400, 180)
    put(c, 60, 500, 10, "표 1. 예산 현황(단위: 천원)")
    table(c, 60, 485, [120, 120, 120, 120], [["구분", "1분기", "2분기", "합계"], ["인건비", "1,200", "1,350", "2,550"],
                                             ["운영비", "800", "760", "1,560"], ["합계", "2,000", "2,110", "4,110"]])
    put(c, 60, 360, 10.5, "표 아래 문단이다. 표 제목은 그림 캡션이 아니다.")
    put(c, 290, 40, 9, "- 2 -")


def build() -> bytes:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4, invariant=1, pageCompression=0)
    for draw in (page1, page2):
        draw(c)
        c.showPage()
    c.save()
    return buf.getvalue()


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(build())
    print(OUT)
