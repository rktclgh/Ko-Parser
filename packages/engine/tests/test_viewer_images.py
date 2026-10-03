import gc
import io
import math
import threading

import pypdfium2 as pdfium
import pypdfium2.internal as pdfium_i
import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from ko_parser.errors import ParseError
from ko_parser.formats.pdf.extract import PDFIUM_LOCK
from ko_parser.viewer import DEFAULT_DPI, render_page_images
from ko_parser.viewer.images import _MAX_PIXELS, _MAX_SIDE


def make_pdf(pages: int = 2, **kw) -> bytes:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(595.0, 842.0), invariant=1, pageCompression=0, **kw)
    for _ in range(pages):
        c.rect(72, 72, 200, 100, fill=1)
        c.showPage()
    c.save()
    return buf.getvalue()


def live_pdfium_objects() -> int:
    # ObjectTracker는 pypdfium2 비공개 API: pyproject의 pypdfium2>=5.8,<6 범위에 기댄다
    return sum(len(refs) for refs in pdfium_i.ObjectTracker.values())


def test_jpeg_per_page_at_requested_dpi():
    data = make_pdf()
    gc.collect()
    before = live_pdfium_objects()
    images = render_page_images(data, "a.pdf", dpi=72)
    assert live_pdfium_objects() == before  # 문서·쪽·비트맵을 모두 닫았다(번들 글꼴 등록이 미뤄지지 않는다)
    assert sorted(images) == [1, 2]
    image = Image.open(io.BytesIO(images[1]))
    assert (image.format, image.mode, image.size) == ("JPEG", "RGB", (595, 842))  # 595·842pt × 72/72
    hi = Image.open(io.BytesIO(render_page_images(make_pdf(1), "a.pdf", dpi=144)[1]))
    assert (hi.format, hi.size) == ("JPEG", (1190, 1684))  # × 144/72


def test_render_waits_for_the_package_pdfium_lock():
    """추출과 같은 패키지 잠금을 쓴다. 다른 스레드가 잡고 있으면 렌더는 기다린다."""
    data = make_pdf(1)
    done = threading.Event()
    worker = threading.Thread(target=lambda: (render_page_images(data, "a.pdf", dpi=36), done.set()))
    with PDFIUM_LOCK:
        worker.start()
        assert not done.wait(0.3)
    worker.join(10)
    assert done.is_set()


def test_default_dpi_is_110():
    assert DEFAULT_DPI == 110
    assert Image.open(io.BytesIO(render_page_images(make_pdf(1), "a.pdf")[1])).size == (910, 1287)  # 595·842pt × 110/72, 올림


def test_rotated_page_renders_as_displayed():
    doc = pdfium.PdfDocument.new()
    doc.new_page(595, 842).set_rotation(90)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    assert Image.open(io.BytesIO(render_page_images(buf.getvalue(), "r.pdf", dpi=72)[1])).size == (842, 595)


def test_encrypted_or_corrupt_is_parse_error():
    with pytest.raises(ParseError, match="encrypted PDF"):
        render_page_images(make_pdf(encrypt="secret"), "암호.pdf")
    with pytest.raises(ParseError, match="invalid PDF"):
        render_page_images(b"%PDF-1.4\n", "깨짐.pdf")
    with pytest.raises(ValueError):
        render_page_images(make_pdf(1), "a.pdf", dpi=0)


def _fail(*args, **kwargs):
    raise pdfium.PdfiumError("boom")


@pytest.mark.parametrize("target", [(pdfium.PdfDocument, "__getitem__"), (pdfium.PdfPage, "render")])
def test_page_failure_is_parse_error_and_releases_everything(monkeypatch, target):
    data = make_pdf(1)
    gc.collect()
    before = live_pdfium_objects()
    monkeypatch.setattr(*target, _fail)
    with pytest.raises(ParseError, match="invalid PDF page: boom") as info:
        render_page_images(data, "a.pdf", dpi=36)
    assert info.value.location == "a.pdf:1"
    assert PDFIUM_LOCK.acquire(blocking=False)  # 잠금을 놓았다
    PDFIUM_LOCK.release()
    assert live_pdfium_objects() == before


def test_huge_page_is_capped_at_max_pixels():
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(14400.0, 7200.0), invariant=1, pageCompression=0)
    c.rect(72, 72, 200, 100, fill=1)
    c.showPage()
    c.save()
    image = Image.open(io.BytesIO(render_page_images(buf.getvalue(), "big.pdf")[1]))
    width, height = image.size
    assert width * height <= _MAX_PIXELS
    assert width * height > _MAX_PIXELS * 0.99  # 필요한 만큼만 줄였다
    assert math.isclose(width, height * 2, abs_tol=2)  # 가로세로 비율 유지(±1px씩)


def test_tall_page_is_capped_at_max_side():
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(72.0, 14400.0), invariant=1, pageCompression=0)
    c.rect(10, 72, 50, 100, fill=1)
    c.showPage()
    c.save()
    image = Image.open(io.BytesIO(render_page_images(buf.getvalue(), "tall.pdf", dpi=600)[1]))
    width, height = image.size
    assert max(width, height) <= _MAX_SIDE
    assert height > _MAX_SIDE * 0.99  # 필요한 만큼만 줄였다
    assert math.isclose(width, height / 200, abs_tol=2)  # 가로세로 비율 유지
