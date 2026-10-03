"""pypdfium2로 쪽마다 글자·그림을 읽는다.

PDFium은 스레드 안전하지 않다. 문서가 달라도 동시에 부르면 프로세스가 죽으므로 PDFium을 부르는 동안(문서를
열고 모두 닫을 때까지) 패키지에 하나뿐인 PDFIUM_LOCK을 잡는다. 쪽 그림 렌더러도 같은 잠금을 쓴다.
글자 상자는 글꼴 사전의 너비(/W)·ascent·descent와 글자 원점으로 계산한다. PDFium의 글리프 상자는
미임베드 글꼴이면 OS의 대체 글꼴에 따라 달라지므로 그 정보가 없을 때만 쓴다.
미임베드 글꼴은 PDFium이 시스템 글꼴로 대신 그린다. 한글 글꼴이 없는 컴퓨터(글꼴 없는 Linux 등)에서는 한 글자짜리
글자 객체가 텍스트에서 통째로 빠지므로 조용히 버리지 않고 ParseError로 알린다(_check_dropped_text).
"""

import ctypes
import math
import re
import threading
import warnings
from collections.abc import Iterator
from dataclasses import dataclass

with warnings.catch_warnings():  # pypdfium2_raw는 import할 때 버전 파일을 인코딩 없이 연다(EncodingWarning)
    warnings.simplefilter("ignore", EncodingWarning)
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

from ...errors import ParseError

Box = tuple[float, float, float, float]  # left, bottom, right, top (PDF 쪽 좌표)
Axes = tuple[int, int]  # (진행 방향, 줄 아래 방향). 보이는 쪽의 +x·+y·−x·−y = 0·1·2·3
UPRIGHT: Axes = (0, 1)

PDFIUM_LOCK = threading.Lock()  # PDFium 호출 전체를 줄 세운다(문서가 달라도)

_BOLD_NAME = re.compile(r"bold|black|heavy", re.IGNORECASE)
_BOLD_WEIGHT = 600


@dataclass(frozen=True, slots=True)
class Char:
    """글자 하나. 상자는 보이는 쪽 기준 0~1(원점 왼쪽 위, 회전 보정 후), size는 실제 크기(pt).
    baseline은 줄 아래 방향 축(axes[1]) 위 글자 원점의 위치(0~1). 바로 선 글자면 보이는 y와 같다."""

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
    axes: Axes = UPRIGHT  # 읽는 방향: 회전한 쪽·음수 Tf·거울 행렬이면 바로 선 글자와 다르다


@dataclass(frozen=True, slots=True)
class PageText:
    page: int
    width_pt: float  # 회전 보정 후
    height_pt: float
    rotation: int
    chars: tuple[Char, ...]
    image_coverage: tuple[float, ...]  # 그림마다 쪽 면적 대비 비율


def open_pdf(data: bytes, name: str) -> pdfium.PdfDocument:
    """암호화·손상 PDF는 ParseError. 부르는 쪽이 PDFIUM_LOCK을 잡고 있어야 한다."""
    try:
        return pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        if getattr(exc, "err_code", None) == pdfium_c.FPDF_ERR_PASSWORD:
            raise ParseError("encrypted PDF", name) from None
        raise ParseError(f"invalid PDF: {exc}", name) from None


def extract_pages(data: bytes, name: str) -> tuple[PageText, ...]:
    """쪽이 없거나(PDFium은 대개 열기부터 실패) 쪽을 읽지 못하면 ParseError."""
    with PDFIUM_LOCK:
        pdf = open_pdf(data, name)
        try:
            if len(pdf) == 0:
                raise ParseError("PDF has no pages", name)
            pages = []
            for index in range(len(pdf)):
                location = f"{name}:{index + 1}"
                try:
                    pages.append(_page(pdf, index, location))
                except pdfium.PdfiumError as exc:
                    raise ParseError(f"invalid PDF page: {exc}", location) from None
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


def _axis(dx: float, dy: float) -> int:
    """보이는 쪽(원점 왼쪽 위) 벡터의 주된 방향: +x·+y·−x·−y = 0·1·2·3."""
    if abs(dx) >= abs(dy):
        return 0 if dx >= 0 else 2
    return 1 if dy > 0 else 3


def _reading_axes(advance: tuple[float, float], up: tuple[float, float], box: Box, rotation: int) -> Axes:
    """PDF 쪽 좌표의 진행 벡터·글자 위쪽 벡터 → 보이는 쪽의 (진행 방향, 줄 아래 방향). 둘이 나란하면(기울임이
    지나친 행렬) 진행 방향의 시계 방향 90°를 줄 아래로 본다. 진행 벡터가 0이면 바로 선 글자."""
    left, bottom, right, top = box
    w, h = (right - left, top - bottom) if rotation in (0, 180) else (top - bottom, right - left)
    x0, y0 = normalize_point(left, bottom, box, rotation)

    def visible(vx: float, vy: float) -> tuple[float, float]:
        x1, y1 = normalize_point(left + vx, bottom + vy, box, rotation)
        return (x1 - x0) * w, (y1 - y0) * h

    ax, ay = visible(*advance)
    if ax == 0 and ay == 0:
        return UPRIGHT
    along = _axis(ax, ay)
    ux, uy = visible(*up)
    down = _axis(-ux, -uy) if (ux or uy) else (along + 1) % 4
    return (along, down) if down % 2 != along % 2 else (along, (along + 1) % 4)


def _on_axis(x: float, y: float, axis: int) -> float:
    """보이는 쪽 점(0~1)의 axis 방향 좌표(0~1)."""
    return (x, y, 1 - x, 1 - y)[axis]


def decode_unicode(code: int, following: int | None) -> tuple[str, bool, bool]:
    """(글자, 매핑 실패, 다음 코드를 함께 썼는지). UTF-16 대리쌍은 합치고 짝 없는 대리 코드는 매핑 실패."""
    if 0xD800 <= code < 0xDC00 and following is not None and 0xDC00 <= following < 0xE000:
        return chr(0x10000 + ((code - 0xD800) << 10) + (following - 0xDC00)), False, True
    if code in (0, 0xFFFD) or 0xD800 <= code < 0xE000:
        return "\ufffd", True, False
    return chr(code), False, False


def _address(handle: object) -> int:
    return ctypes.cast(handle, ctypes.c_void_p).value or 0


def _page(pdf: pdfium.PdfDocument, index: int, location: str) -> PageText:
    page = pdf[index]
    try:
        width, height = page.get_size()
        rotation = page.get_rotation()
        box = page.get_bbox()
        left, bottom, right, top = box
        if not (right > left and top > bottom):  # 예: CropBox가 MediaBox 밖(0×0). 좌표를 0~1로 바꿀 수 없다
            raise ParseError("PDF page has an empty box", location)
        textpage = page.get_textpage()
        seen: set[int] = set()
        try:
            chars = tuple(_chars(textpage, box, rotation, seen))
        finally:
            textpage.close()
        _check_dropped_text(pdf, page, seen, location)
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


def _check_dropped_text(pdf: pdfium.PdfDocument, page: pdfium.PdfPage, seen: set[int], location: str) -> None:
    """PDFium의 텍스트 쪽은 상자 너비가 0에 가까운 글자 객체를 건너뛴다. 객체 상자는 글리프 외곽으로 재므로 미임베드
    글꼴의 글리프를 이 컴퓨터의 어떤 글꼴에서도 찾지 못하면 한 글자짜리 객체가 글자째 빠진다(실측: 한글 글꼴 없는
    Linux). 공백 한 칸짜리 객체도 같은 이유로 모든 OS에서 빠지므로, 텍스트 쪽이 건너뛴 객체의 미임베드 글꼴이
    이 컴퓨터에서 한글을 그리지 못할 때만 ParseError(seen은 텍스트 쪽에 글자가 있는 객체 주소)."""
    draws: dict[int, bool] = {}
    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_TEXT]):
        if _address(obj.raw) in seen:
            continue
        font = pdfium_c.FPDFTextObj_GetFont(obj.raw)
        if not font or pdfium_c.FPDFFont_GetIsEmbedded(font) == 1:
            continue
        key = _address(font)
        if key not in draws:
            draws[key] = _draws_hangul(pdf, font)
        if not draws[key]:
            raise ParseError("PDF text uses a non-embedded font that has no Hangul glyphs on this system; "
                             "install a Korean font (e.g. fonts-noto-cjk)", location)


def _draws_hangul(pdf: pdfium.PdfDocument, font: object) -> bool:
    """이 글꼴로 '가'를 그리면 상자가 생기는가(PDFium이 대신 쓸 한글 글리프를 찾았는가). 쪽에 넣지 않는 임시 객체."""
    obj = pdfium_c.FPDFPageObj_CreateTextObj(pdf.raw, font, ctypes.c_float(1.0))
    if not obj:
        return False
    try:
        text = ctypes.create_string_buffer("가\0".encode("utf-16-le"))
        if not pdfium_c.FPDFText_SetText(obj, ctypes.cast(text, ctypes.POINTER(pdfium_c.FPDF_WCHAR))):
            return False
        left, bottom, right, top = (ctypes.c_float() for _ in range(4))
        return bool(pdfium_c.FPDFPageObj_GetBounds(obj, left, bottom, right, top)) and right.value > left.value
    finally:
        pdfium_c.FPDFPageObj_Destroy(obj)


def _chars(textpage: pdfium.PdfTextPage, box: Box, rotation: int, seen: set[int]) -> Iterator[Char]:
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
        seen.add(key)
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
        has_width = pdfium_c.FPDFFont_GetGlyphWidth(font, ord(text[0]), ctypes.c_float(abs(font_size)), width)
        sign = math.copysign(1.0, font_size)  # Tf가 음수면 진행·위쪽이 모두 뒤집힌다(180°)
        axes = _reading_axes((m.a * sign, m.b * sign), (m.c * sign, m.d * sign), box, rotation)
        if has_width and width.value > 0 and ascent > descent:
            # Tf가 음수면 글자가 원점에서 왼쪽·아래로 뒤집혀 그려진다: 진행 폭도 높이처럼 Tf 부호를 따른다
            advance = math.copysign(width.value, font_size)
            corners = [(ox.value + m.a * x + m.c * y, oy.value + m.b * x + m.d * y)
                       for x in (0.0, advance) for y in (descent * font_size, ascent * font_size)]
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
                   baseline=_on_axis(*normalize_point(ox.value, oy.value, box, rotation), axes[1]),
                   size=abs(font_size) * math.hypot(m.c, m.d),  # Tf가 음수면 글자가 뒤집힐 뿐 크기는 양수
                   bold=weight >= _BOLD_WEIGHT or bold_name or mode == pdfium_c.FPDF_TEXTRENDERMODE_FILL_STROKE,
                   invisible=mode == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE, unmapped=unmapped, axes=axes)


_MAX_FORM_DEPTH = 15


def _clip_box(obj: pdfium.PdfObject, matrix: pdfium.PdfMatrix | None) -> Box | None:
    """객체 클리핑 경로들의 교집합 상자(쪽 좌표, 경로마다 점들의 외접 상자). 클리핑이 없거나 읽지 못하면 None,
    읽지 못한 경로 하나는 건너뛴다(넓게 잡는 쪽으로). 폼 안 객체의 경로는 폼 좌표라 matrix로 옮긴다(실측)."""
    clip = pdfium_c.FPDFPageObj_GetClipPath(obj)
    if not clip:
        return None
    out: Box | None = None
    for path in range(max(pdfium_c.FPDFClipPath_CountPaths(clip), 0)):
        points = []
        for index in range(max(pdfium_c.FPDFClipPath_CountPathSegments(clip, path), 0)):
            segment = pdfium_c.FPDFClipPath_GetPathSegment(clip, path, index)
            x, y = ctypes.c_float(), ctypes.c_float()
            if not (segment and pdfium_c.FPDFPathSegment_GetPoint(segment, x, y)):
                points = []
                break
            points.append(matrix.on_point(x.value, y.value) if matrix is not None else (x.value, y.value))
        if not points:
            continue
        box = (min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points),
               max(p[1] for p in points))
        out = box if out is None else _intersect(out, box)
    return out


def _intersect(a: Box, b: Box | None) -> Box:
    """겹치지 않으면 폭이나 높이가 음수인 상자(면적 계산에서 0)."""
    if b is None:
        return a
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def _image_boxes(page: pdfium.PdfPage, form: pdfium.PdfObject | None = None, matrix: pdfium.PdfMatrix | None = None,
                 clip: Box | None = None, depth: int = 0) -> Iterator[Box]:
    """그림마다 보이는 부분의 쪽 좌표 상자(그림 상자 ∩ 클리핑 상자). 폼 XObject 안 객체의 get_bounds()·클리핑
    경로는 폼 좌표라(실측) 폼 행렬을 거쳐 옮긴다. 폼 객체의 클리핑은 폼 안 객체에도 적용된다.
    폼 상자 자체는 그림으로 보지 않는다(쪽 전체 서식 폼 안의 작은 로고가 쪽 전체 그림이 되지 않게)."""
    for obj in page.get_objects(max_depth=1, form=form):
        if obj.type not in (pdfium_c.FPDF_PAGEOBJ_IMAGE, pdfium_c.FPDF_PAGEOBJ_FORM):
            continue
        own = _clip_box(obj, matrix)
        visible = clip if own is None else _intersect(own, clip)
        if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
            bounds = obj.get_bounds()
            yield _intersect(matrix.on_rect(*bounds) if matrix is not None else bounds, visible)
        elif depth < _MAX_FORM_DEPTH:
            inner = obj.get_matrix() if matrix is None else obj.get_matrix().multiply(matrix)
            yield from _image_boxes(page, obj, inner, visible, depth + 1)


def _image_coverage(page: pdfium.PdfPage, box: Box) -> Iterator[float]:
    """그림마다 쪽 상자 안에 보이는 면적 / 쪽 면적."""
    left, bottom, right, top = box
    area = (right - left) * (top - bottom)
    for l, b, r, t in _image_boxes(page):
        overlap = max(0.0, min(r, right) - max(l, left)) * max(0.0, min(t, top) - max(b, bottom))
        yield min(1.0, overlap / area)
