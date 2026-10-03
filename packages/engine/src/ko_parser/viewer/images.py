"""뷰어용 쪽 이미지: pypdfium2로 렌더해 Pillow JPEG(품질 80)로 만든다.

PDFium은 스레드 안전하지 않고 open_pdf는 번들 글꼴 등록(PDFium 다시 초기화)을 할 수 있으므로 문서를 열고 닫을
때까지 추출과 같은 PDFIUM_LOCK을 잡는다. 쪽·비트맵은 쓰자마자 닫아 살아 있는 pypdfium2 객체를 남기지 않는다.
"""

import io
import math

from ..errors import ParseError
from ..formats.pdf.extract import PDFIUM_LOCK, open_pdf, pdfium

DEFAULT_DPI = 110
_JPEG_QUALITY = 80
_MAX_PIXELS = 25_000_000  # 쪽 그림 하나의 화소 상한(약 5000×5000). 넘는 쪽은 배율을 낮춘다


def render_page_images(data: bytes, name: str, dpi: int = DEFAULT_DPI) -> dict[int, bytes]:
    """{쪽 번호: JPEG 바이트}. 쪽 회전(/Rotate)은 렌더에 반영되고 _MAX_PIXELS를 넘는 쪽은 비율을 지켜 줄인다.
    쪽이 없거나 암호화·손상 PDF는 ParseError."""
    if dpi < 1:
        raise ValueError("dpi must be >= 1")
    images: dict[int, bytes] = {}
    with PDFIUM_LOCK:
        pdf = open_pdf(data, name)
        try:
            if len(pdf) == 0:
                raise ParseError("PDF has no pages", name)
            for index in range(len(pdf)):
                location = f"{name}:{index + 1}"
                page = bitmap = None
                try:
                    page = pdf[index]
                    width, height = page.get_size()
                    scale = dpi / 72
                    if math.ceil(width * scale) * math.ceil(height * scale) > _MAX_PIXELS:
                        scale = math.sqrt(_MAX_PIXELS / (width * height))
                        while math.ceil(width * scale) * math.ceil(height * scale) > _MAX_PIXELS:
                            scale *= 0.999  # pypdfium2가 화소 수를 올림하므로 넘치면 조금 더 줄인다
                    bitmap = page.render(scale=scale)
                    image = bitmap.to_pil().convert("RGB")  # 복사본: 비트맵을 닫아도 남는다
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG", quality=_JPEG_QUALITY)
                    images[index + 1] = buf.getvalue()
                except pdfium.PdfiumError as exc:
                    raise ParseError(f"invalid PDF page: {exc}", location) from None
                finally:
                    try:
                        if bitmap is not None:
                            bitmap.close()
                    finally:
                        if page is not None:
                            page.close()
        finally:
            pdf.close()
    return images
