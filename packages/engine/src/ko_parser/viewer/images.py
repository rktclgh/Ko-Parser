"""뷰어용 쪽 이미지: pypdfium2로 렌더해 Pillow JPEG(품질 80)로 만든다.

PDFium은 스레드 안전하지 않고 open_pdf는 번들 글꼴 등록(PDFium 다시 초기화)을 할 수 있으므로 문서를 열고 닫을
때까지 추출과 같은 PDFIUM_LOCK을 잡는다. 쪽·비트맵은 쓰자마자 닫아 살아 있는 pypdfium2 객체를 남기지 않는다.
"""

import io

from ..errors import ParseError
from ..formats.pdf.extract import PDFIUM_LOCK, open_pdf, pdfium

DEFAULT_DPI = 110
_JPEG_QUALITY = 80


def render_page_images(data: bytes, name: str, dpi: int = DEFAULT_DPI) -> dict[int, bytes]:
    """{쪽 번호: JPEG 바이트}. 쪽 회전(/Rotate)은 렌더에 반영된다. 암호화·손상 PDF는 ParseError."""
    if dpi < 1:
        raise ValueError("dpi must be >= 1")
    images: dict[int, bytes] = {}
    with PDFIUM_LOCK:
        pdf = open_pdf(data, name)
        try:
            for index in range(len(pdf)):
                page = pdf[index]
                bitmap = None
                try:
                    bitmap = page.render(scale=dpi / 72)
                    image = bitmap.to_pil().convert("RGB")  # 복사본: 비트맵을 닫아도 남는다
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG", quality=_JPEG_QUALITY)
                    images[index + 1] = buf.getvalue()
                except pdfium.PdfiumError as exc:
                    raise ParseError(f"cannot render page: {exc}", f"{name}:{index + 1}") from None
                finally:
                    if bitmap is not None:
                        bitmap.close()
                    page.close()
        finally:
            pdf.close()
    return images
