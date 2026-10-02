"""pypdfium2로 쪽마다 글자·그림을 읽는다.

PDFium은 스레드 안전하지 않으므로 문서 하나는 한 스레드에서 처리한다.
글자 상자는 글꼴 사전의 너비(/W)·ascent·descent와 글자 원점으로 계산한다. PDFium의 글리프 상자는
미임베드 글꼴이면 OS의 대체 글꼴에 따라 달라지므로 그 정보가 없을 때만 쓴다.
"""

import ctypes
import math
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass

with warnings.catch_warnings():  # pypdfium2_raw는 import할 때 버전 파일을 인코딩 없이 연다(EncodingWarning)
    warnings.simplefilter("ignore", EncodingWarning)
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

from ...errors import ParseError

Box = tuple[float, float, float, float]  # left, bottom, right, top (PDF 쪽 좌표)

_BOLD_NAME = re.compile(r"bold|black|heavy", re.IGNORECASE)
_BOLD_WEIGHT = 600


@dataclass(frozen=True, slots=True)
class Char:
    """글자 하나. 좌표는 쪽 기준 0~1(원점 왼쪽 위, 회전 보정 후), size는 실제 크기(pt)."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    baseline: float
    size: float
    bold: bool = False
    invisible: bool = False  # 렌더 모드 3
    unmapped: bool = False  # 유니코드 0·U+FFFD·매핑 오류(text는 U+FFFD)


@dataclass(frozen=True, slots=True)
class PageText:
    page: int
    width_pt: float  # 회전 보정 후
    height_pt: float
    rotation: int
    chars: tuple[Char, ...]
    image_coverage: tuple[float, ...]  # 그림마다 쪽 면적 대비 비율


def open_pdf(data: bytes, name: str) -> pdfium.PdfDocument:
    """암호화·손상 PDF는 ParseError."""
    try:
        return pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        if getattr(exc, "err_code", None) == pdfium_c.FPDF_ERR_PASSWORD:
            raise ParseError("encrypted PDF", name) from None
        raise ParseError(f"invalid PDF: {exc}", name) from None


def extract_pages(data: bytes, name: str) -> tuple[PageText, ...]:
    """쪽이 없거나(PDFium은 대개 열기부터 실패) 쪽을 읽지 못하면 ParseError."""
    pdf = open_pdf(data, name)
    try:
        if len(pdf) == 0:
            raise ParseError("PDF has no pages", name)
        pages = []
        for index in range(len(pdf)):
            try:
                pages.append(_page(pdf, index))
            except pdfium.PdfiumError as exc:
                raise ParseError(f"invalid PDF page: {exc}", f"{name}:{index + 1}") from None
        return tuple(pages)
    finally:
        pdf.close()


def normalize_point(x: float, y: float, box: Box, rotation: int) -> tuple[float, float]:
    """PDF 쪽 좌표 → 보이는 쪽 기준 0~1(원점 왼쪽 위). rotation은 /Rotate(시계 방향)."""
    left, bottom, right, top = box
    u, v = (x - left) / (right - left), (top - y) / (top - bottom)
    match rotation:
        case 90:
            return 1 - v, u
        case 180:
            return 1 - u, 1 - v
        case 270:
            return v, 1 - u
        case _:
            return u, v


def decode_unicode(code: int, following: int | None) -> tuple[str, bool, bool]:
    """(글자, 매핑 실패, 다음 코드를 함께 썼는지). UTF-16 대리쌍은 합치고 짝 없는 대리 코드는 매핑 실패."""
    if 0xD800 <= code < 0xDC00 and following is not None and 0xDC00 <= following < 0xE000:
        return chr(0x10000 + ((code - 0xD800) << 10) + (following - 0xDC00)), False, True
    if code in (0, 0xFFFD) or 0xD800 <= code < 0xE000:
        return "\ufffd", True, False
    return chr(code), False, False


def _address(handle: object) -> int:
    return ctypes.cast(handle, ctypes.c_void_p).value or 0


def _page(pdf: pdfium.PdfDocument, index: int) -> PageText:
    page = pdf[index]
    try:
        width, height = page.get_size()
        rotation = page.get_rotation()
        box = page.get_bbox()
        textpage = page.get_textpage()
        try:
            chars = tuple(_chars(textpage, box, rotation))
        finally:
            textpage.close()
        return PageText(page=index + 1, width_pt=width, height_pt=height, rotation=rotation, chars=chars,
                        image_coverage=tuple(_image_coverage(page, box)))
    finally:
        page.close()


class _Fonts:
    """글꼴 핸들별 (1pt당 ascent, 1pt당 descent, 이름이 굵은 글꼴인지)."""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[float, float, bool]] = {}

    def get(self, font: object) -> tuple[float, float, bool]:
        key = _address(font)
        if key not in self._cache:
            ascent, descent, one = ctypes.c_float(), ctypes.c_float(), ctypes.c_float(1.0)
            if not (pdfium_c.FPDFFont_GetAscent(font, one, ascent) and pdfium_c.FPDFFont_GetDescent(font, one, descent)):
                ascent.value = descent.value = 0.0
            length = pdfium_c.FPDFFont_GetBaseFontName(font, None, 0)
            name = ctypes.create_string_buffer(max(length, 1))
            pdfium_c.FPDFFont_GetBaseFontName(font, name, length)
            self._cache[key] = (ascent.value, descent.value, bool(_BOLD_NAME.search(name.value.decode("latin-1"))))
        return self._cache[key]


def _chars(textpage: pdfium.PdfTextPage, box: Box, rotation: int) -> Iterator[Char]:
    fonts = _Fonts()
    modes: dict[int, int] = {}
    count = textpage.count_chars()
    skip = False
    for i in range(count):
        if skip:
            skip = False
            continue
        if pdfium_c.FPDFText_IsGenerated(textpage, i) == 1:  # PDFium이 끼운 공백·줄바꿈
            continue
        obj = pdfium_c.FPDFText_GetTextObject(textpage, i)
        if not obj:
            continue
        code = pdfium_c.FPDFText_GetUnicode(textpage, i)
        high = 0xD800 <= code < 0xDC00 and i + 1 < count  # UTF-16 대리쌍의 앞 절반
        following = pdfium_c.FPDFText_GetUnicode(textpage, i + 1) if high else None
        text, unmapped, skip = decode_unicode(code, following)
        unmapped = unmapped or pdfium_c.FPDFText_HasUnicodeMapError(textpage, i) == 1
        if unmapped:
            text = "\ufffd"
        key = _address(obj)
        if key not in modes:
            modes[key] = pdfium_c.FPDFTextObj_GetTextRenderMode(obj)
        font = pdfium_c.FPDFTextObj_GetFont(obj)
        ascent, descent, bold_name = fonts.get(font)
        font_size = pdfium_c.FPDFText_GetFontSize(textpage, i)
        m = pdfium_c.FS_MATRIX()
        pdfium_c.FPDFText_GetMatrix(textpage, i, m)
        ox, oy = ctypes.c_double(), ctypes.c_double()
        pdfium_c.FPDFText_GetCharOrigin(textpage, i, ox, oy)
        width = ctypes.c_float()
        # 너비는 유니코드 → 글자 코드 역매핑으로 찾는다. 같은 유니코드에 글자 코드가 여럿이면 하나만 보므로
        # 너비가 조금 다를 수 있다(bbox·공백 판단에만 영향, 글자는 그대로).
        has_width = pdfium_c.FPDFFont_GetGlyphWidth(font, ord(text[0]), ctypes.c_float(font_size), width)
        if has_width and width.value > 0 and ascent > descent:
            corners = [(ox.value + m.a * x + m.c * y, oy.value + m.b * x + m.d * y)
                       for x in (0.0, width.value) for y in (descent * font_size, ascent * font_size)]
        else:  # 글꼴 정보가 없으면 PDFium의 느슨한 상자(대체 글꼴에 따라 달라질 수 있다)
            rect = pdfium_c.FS_RECTF()
            pdfium_c.FPDFText_GetLooseCharBox(textpage, i, rect)
            corners = [(rect.left, rect.bottom), (rect.right, rect.top)]
        points = [normalize_point(x, y, box, rotation) for x, y in corners]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        if not (0 <= cx <= 1 and 0 <= cy <= 1):  # 쪽 밖 글자는 보이지 않는다
            continue
        weight = pdfium_c.FPDFText_GetFontWeight(textpage, i)
        mode = modes[key]
        yield Char(text=text, x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys),
                   baseline=normalize_point(ox.value, oy.value, box, rotation)[1],
                   size=abs(font_size) * math.hypot(m.c, m.d),  # Tf가 음수면 글자가 뒤집힐 뿐 크기는 양수
                   bold=weight >= _BOLD_WEIGHT or bold_name or mode == pdfium_c.FPDF_TEXTRENDERMODE_FILL_STROKE,
                   invisible=mode == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE, unmapped=unmapped)


_MAX_FORM_DEPTH = 15


def _image_boxes(page: pdfium.PdfPage, form: pdfium.PdfObject | None = None, matrix: pdfium.PdfMatrix | None = None,
                 depth: int = 0) -> Iterator[Box]:
    """그림마다 쪽 좌표 상자. 폼 XObject 안 객체의 get_bounds()는 폼 좌표라(실측) 폼 행렬을 거쳐 옮긴다.
    폼 상자 자체는 그림으로 보지 않는다(쪽 전체 서식 폼 안의 작은 로고가 쪽 전체 그림이 되지 않게)."""
    for obj in page.get_objects(max_depth=1, form=form):
        if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
            bounds = obj.get_bounds()
            yield matrix.on_rect(*bounds) if matrix is not None else bounds
        elif obj.type == pdfium_c.FPDF_PAGEOBJ_FORM and depth < _MAX_FORM_DEPTH:
            inner = obj.get_matrix() if matrix is None else obj.get_matrix().multiply(matrix)
            yield from _image_boxes(page, obj, inner, depth + 1)


def _image_coverage(page: pdfium.PdfPage, box: Box) -> Iterator[float]:
    """그림마다 쪽 상자 안에 든 면적 / 쪽 면적."""
    left, bottom, right, top = box
    area = (right - left) * (top - bottom)
    for l, b, r, t in _image_boxes(page):
        overlap = max(0.0, min(r, right) - max(l, left)) * max(0.0, min(t, top) - max(b, bottom))
        yield min(1.0, overlap / area)
