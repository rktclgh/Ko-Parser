"""PdfParser의 스캔 쪽 OCR: scanned 쪽만 읽고, 텍스트 레이어가 우선이며, 블록은 윗변 순서로 합쳐진다.
그림은 OS 글꼴에 기대지 않으려고 ko-parser-fonts 글꼴(Pillow)로 그려 reportlab PDF에 넣는다."""

import importlib.machinery
import io
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

import ko_parser_fonts
from ko_parser import models
from ko_parser.errors import OcrUnavailable
from ko_parser.formats.pdf import PdfParser, ocr, scan

FONT = "HYGothic-Medium"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))
NOTO = str(ko_parser_fonts.font_dir() / ko_parser_fonts.FONT_FILE)
PX = 200 / 72  # 그림 화소 / pt
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pdf" / "inputs"
BODY = [(40, 60, 14, "스캔한 쪽의 글자를 읽는다."), (40, 82, 14, "두 줄로 된 문단이다."),
        (40, 140, 14, "다음 문단은 한 줄이다.")]


def text_image(width: float, height: float, lines) -> Image.Image:
    """보이는 쪽 크기(pt)의 흰 그림에 (왼쪽 pt, 윗변 pt, 크기 pt, 글자) 줄을 그린다(200 DPI)."""
    img = Image.new("L", (round(width * PX), round(height * PX)), 255)
    draw = ImageDraw.Draw(img)
    for x, top, size, s in lines:
        draw.text((x * PX, top * PX), s, font=ImageFont.truetype(NOTO, round(size * PX)), fill=0)
    return img


def put(c: Canvas, x: float, y: float, size: float, s: str) -> None:
    c.setFont(FONT, size)
    c.drawString(x, y, s)


def scanned_page(c: Canvas, width: float, height: float, lines, visible=()) -> None:
    """쪽 전체 그림 + 보이는 글자(왼쪽, 기준선 y(PDF 좌표), 크기, 글자)."""
    c.drawImage(ImageReader(text_image(width, height, lines)), 0, 0, width=width, height=height)
    for x, y, size, s in visible:
        put(c, x, y, size, s)


def pdf(*pages, size=(300.0, 400.0)) -> bytes:
    """pages: Canvas를 받아 한 쪽을 그리는 함수들."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=size, invariant=1, pageCompression=0)
    for draw in pages:
        draw(c)
        c.showPage()
    c.save()
    return buf.getvalue()


def require_models(*names: str) -> None:
    """실제 모델 파일이 필요한 테스트: 찾을 수 없으면 건너뛰고, KO_PARSER_CI_REQUIRE_MODELS=1이면 실패한다."""
    missing = [name for name in names if models.find(name) is None]
    if missing:
        if os.environ.get("KO_PARSER_CI_REQUIRE_MODELS") == "1":
            pytest.fail(f"model files not found: {missing}; run `ko-parser models fetch`")
        pytest.skip(f"model files not found: {missing} (ko-parser models fetch)")


def fake_model_files(monkeypatch, tmp_path) -> None:
    """찾기만 되는 가짜 OCR 모델 파일(KO_PARSER_MODEL_DIR, 해시는 틀리다): 설치 확인이 모델 때문에 거짓이 되지 않게."""
    for name in ocr.MODEL_NAMES:
        path = tmp_path / "models" / models.MANIFEST[name].path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a model")
    monkeypatch.setenv("KO_PARSER_MODEL_DIR", str(tmp_path / "models"))


def blocks(data: bytes, parser: PdfParser | None = None) -> list[tuple[str, str, str]]:
    return [(b["kind"], b["text_source"], b["text"]) for b in (parser or PdfParser()).parse(data, "s.pdf").blocks]


def test_scanned_page_text_becomes_ocr_paragraph_blocks():
    pytest.importorskip("onnxruntime")
    require_models(*ocr.MODEL_NAMES)
    parsed = PdfParser().parse(pdf(lambda c: scanned_page(c, 300, 400, BODY)), "s.pdf")
    assert parsed.pages[0].text_layer == "scanned"
    assert [(b["kind"], b["text_source"], b["text"]) for b in parsed.blocks] == [
        ("paragraph", "ocr", "스캔한 쪽의 글자를 읽는다.\n두 줄로 된 문단이다."),
        ("paragraph", "ocr", "다음 문단은 한 줄이다.")]
    first = parsed.blocks[0]
    assert first["state"] == "det" and first["section_path"] == () and 0.4 <= first["confidence"] <= 0.5
    box = first["locator"]["bbox"]  # 그린 자리: 왼쪽 40pt(0.133), 윗변 60pt(0.15), 두 줄 아래 약 100pt(0.25)
    assert abs(box["x0"] - 40 / 300) <= 0.02 and abs(box["y0"] - 60 / 400) <= 0.02
    assert abs(box["y1"] - 100 / 400) <= 0.02


def test_only_scanned_pages_are_read(monkeypatch):
    pytest.importorskip("onnxruntime")
    require_models(*ocr.MODEL_NAMES)
    calls = []
    real = ocr.read_lines
    monkeypatch.setattr(ocr, "read_lines", lambda image: calls.append(image.size) or real(image))

    def digital(c):
        put(c, 40, 340, 11, "디지털 쪽의 글자는 텍스트 레이어로 읽는다.")

    data = pdf(digital, lambda c: scanned_page(c, 300, 400, BODY[:1]), digital)
    assert blocks(data) == [("paragraph", "text_layer", "디지털 쪽의 글자는 텍스트 레이어로 읽는다."),
                            ("paragraph", "ocr", "스캔한 쪽의 글자를 읽는다."),
                            ("paragraph", "text_layer", "디지털 쪽의 글자는 텍스트 레이어로 읽는다.")]
    assert calls == [(834, 1112)]  # 둘째 쪽(300×400pt를 200 DPI로, 올림)만


def test_ocr_off_or_not_installed_keeps_text_layer_only(monkeypatch):
    data = pdf(lambda c: scanned_page(c, 300, 400, BODY, visible=[(148, 20, 9, "1")]))
    expected = [("paragraph", "text_layer", "1")]
    monkeypatch.setattr(scan, "render", lambda *a: pytest.fail("OCR을 끄면 쪽을 그리지 않는다"))
    assert blocks(data, PdfParser(ocr=False)) == expected
    monkeypatch.setattr(ocr, "available", lambda: False)
    assert blocks(data, PdfParser()) == expected


def test_ocr_true_without_the_install_fails_at_construction(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setattr(ocr, "_reader", None)
    with pytest.raises(OcrUnavailable, match=r"ko-parser-engine\[ocr\]"):
        PdfParser(ocr=True)
    assert PdfParser().ocr is None and PdfParser(ocr=False).ocr is False  # 자동·끔은 만들 때 확인하지 않는다


def test_auto_mode_does_not_check_the_install_without_scanned_pages(monkeypatch):
    """자동 모드: scanned 쪽이 없는 문서는 OCR 추가 설치를 확인하지 않는다(디지털 문서 파싱에 import 비용 없음)."""
    monkeypatch.setattr(ocr, "available", lambda: pytest.fail("scanned 쪽이 없으면 설치를 확인하지 않는다"))
    parsed = PdfParser().parse((FIXTURES / "report.pdf").read_bytes(), "report.pdf")
    assert {p.text_layer for p in parsed.pages} == {"digital"} and parsed.blocks


@pytest.mark.parametrize("name", ["image_page.pdf", "scanned_invisible.pdf"])
def test_auto_mode_with_a_broken_install_raises_instead_of_falling_back(monkeypatch, name):
    """자동 모드에서 설치는 있는데(available) 읽개를 못 만들면 텍스트 레이어로 조용히 물러나지 않고 OcrUnavailable
    (설정 오류). onnxruntime 없이 돈다: 설치 확인과 읽개 만들기를 바꿔 끼운다."""
    def broken():
        raise OcrUnavailable("OCR models could not be loaded: x; run with --no-ocr")

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "get_reader", broken)
    data = (FIXTURES / name).read_bytes()
    assert [p.text_layer for p in PdfParser(ocr=False).parse(data, name).pages] == ["scanned"]
    monkeypatch.setattr(scan, "render", lambda *a: pytest.fail("깨진 설치는 쪽을 그리기 전에 알린다"))
    with pytest.raises(OcrUnavailable, match="--no-ocr"):
        PdfParser().parse(data, name)


def broken_module(monkeypatch, tmp_path, name: str = "onnxruntime") -> None:
    """설치는 됐는데(찾을 수 있다) import하면 깨지는 모듈(공유 라이브러리를 못 여는 휠 흉내)을 sys.path 맨 앞에 둔다.
    진짜 모듈은 sys.modules에서 잠깐 뺀다(끝나면 되돌린다). onnxruntime 없이 돈다."""
    root = tmp_path / "broken"
    (root / name).mkdir(parents=True)
    (root / name / "__init__.py").write_text(
        'raise ImportError("libstub.so.1: cannot open shared object file")\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(ocr, "_reader", None)
    fake_model_files(monkeypatch, tmp_path)  # 모델은 찾히는데 모듈 import가 깨진 설치


def test_installed_but_broken_module_is_available_and_get_reader_names_it(monkeypatch, tmp_path):
    """available()은 '설치됐는가'(찾기만, import하지 않음): 깨진 설치도 참이고, get_reader()가 모듈과 import 오류를
    담은 OcrUnavailable을 낸다(조용히 텍스트 레이어로 물러나지 않게)."""
    broken_module(monkeypatch, tmp_path)
    assert ocr.available() is True
    with pytest.raises(OcrUnavailable, match=r"onnxruntime .*libstub\.so\.1.*ko-parser-engine\[ocr\].* or run with --no-ocr"):
        ocr.get_reader()
    assert ocr._reader is None


def test_leftover_namespace_module_without_its_api_is_a_broken_install(monkeypatch):
    """덜 지운 onnxruntime(빈 폴더 = 이름공간 패키지)은 찾히고 import도 되지만 InferenceSession이 없다. 읽개 모듈을
    가져오다 난 AttributeError도 날 오류로 새지 않고 OcrUnavailable(오류와 설치·--no-ocr 안내). onnxruntime 없이 돈다."""
    pytest.importorskip("numpy")
    pytest.importorskip("pyclipper")
    require_models(*ocr.MODEL_NAMES)  # 모델 파일까지 맞아야 읽개 모듈에서 깨진다
    leftover = types.ModuleType("onnxruntime")
    leftover.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", None, is_package=True)
    leftover.__path__ = []
    monkeypatch.setitem(sys.modules, "onnxruntime", leftover)
    monkeypatch.delitem(sys.modules, "ko_parser.formats.pdf.ocr.reader", raising=False)  # 읽개 모듈을 다시 가져오게
    monkeypatch.setattr(ocr, "_reader", None)
    assert ocr.available() is True
    with pytest.raises(OcrUnavailable,
                       match=r"onnxruntime.* has no attribute .*ko-parser-engine\[ocr\].* or run with --no-ocr"):
        ocr.get_reader()
    assert ocr._reader is None


def without_model_files(monkeypatch) -> None:
    """OCR 모듈은 있는데 모델 파일을 하나도 찾을 수 없는 상태(받기 전). onnxruntime 없이 돈다."""
    monkeypatch.setattr(models, "find", lambda name: None)
    monkeypatch.setattr(ocr, "MODULES", ())
    monkeypatch.setattr(ocr, "_reader", None)


def test_ocr_without_model_files_is_not_installed(monkeypatch):
    """모델 파일을 찾을 수 없으면 설치가 없는 것과 같다(available 거짓). get_reader()·ocr=True는 models fetch 안내."""
    without_model_files(monkeypatch)
    assert ocr.available() is False
    with pytest.raises(OcrUnavailable, match=r"ocr/det\.onnx not found; run `ko-parser models fetch ocr`"):
        ocr.get_reader()
    with pytest.raises(OcrUnavailable, match=r"models fetch ocr.*--no-ocr"):
        PdfParser(ocr=True)
    assert ocr._reader is None


def test_auto_mode_without_ocr_model_files_behaves_as_not_installed(monkeypatch):
    """자동 모드 + 모델 파일 없음: 파싱은 멈추지 않고 OCR 없이(텍스트 레이어만), 쪽도 그리지 않는다."""
    without_model_files(monkeypatch)
    data = (FIXTURES / "image_page.pdf").read_bytes()
    expected = PdfParser(ocr=False).parse(data, "image_page.pdf").blocks
    monkeypatch.setattr(scan, "render", lambda *a: pytest.fail("OCR을 안 하면 쪽을 그리지 않는다"))
    assert PdfParser().parse(data, "image_page.pdf").blocks == expected


def test_auto_mode_with_a_wrong_ocr_model_file_raises(monkeypatch, tmp_path):
    """자동 모드 + 찾은 모델 파일이 고정한 것과 다름(깨진 설치): 텍스트 레이어로 조용히 물러나지 않고 OcrUnavailable."""
    fake_model_files(monkeypatch, tmp_path)
    monkeypatch.setattr(ocr, "MODULES", ())
    monkeypatch.setattr(ocr, "_reader", None)
    monkeypatch.setattr(scan, "render", lambda *a: pytest.fail("깨진 설치는 쪽을 그리기 전에 알린다"))
    with pytest.raises(OcrUnavailable, match=r"does not match the pinned size and SHA-256.*--no-ocr"):
        PdfParser().parse((FIXTURES / "image_page.pdf").read_bytes(), "image_page.pdf")


def test_module_that_cannot_be_found_is_not_installed(monkeypatch):
    monkeypatch.setattr(ocr, "MODULES", ("no_such_ocr_module", *ocr.MODULES))  # find_spec가 None
    monkeypatch.setattr(ocr, "_reader", None)
    assert ocr.available() is False
    with pytest.raises(OcrUnavailable, match=r"not installed \(missing no_such_ocr_module\)"):
        ocr.get_reader()


def test_module_whose_spec_lookup_fails_is_not_installed(monkeypatch):
    """find_spec 자체가 오류(__spec__ 없는 모듈은 ValueError, 없는 상위 패키지는 ModuleNotFoundError)면 설치 없음."""
    monkeypatch.setitem(sys.modules, "pyclipper", type(sys)("pyclipper"))  # __spec__ = None
    monkeypatch.setattr(ocr, "MODULES", ("pyclipper",))
    assert ocr.available() is False
    monkeypatch.setattr(ocr, "MODULES", ("no_such_parent_pkg.child",))
    assert ocr.available() is False


def test_available_does_not_import_the_runtime():
    """설치 확인은 numpy·onnxruntime·pyclipper를 import하지 않는다(하위 프로세스에서 본다)."""
    code = ("import sys\n"
            "from ko_parser.formats.pdf import ocr\n"
            "ocr.available()\n"
            "print(sorted(m for m in ('numpy', 'onnxruntime', 'pyclipper') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize("name", ["image_page.pdf", "scanned_invisible.pdf"])
def test_auto_mode_with_a_broken_module_raises(monkeypatch, tmp_path, name):
    """자동 모드 + 설치는 됐는데 import가 깨지는 onnxruntime: scanned 쪽이 있으면 텍스트 레이어로 조용히 물러나지
    않고 OcrUnavailable(설정 오류)."""
    broken_module(monkeypatch, tmp_path)
    with pytest.raises(OcrUnavailable, match=r"onnxruntime .*libstub"):
        PdfParser().parse((FIXTURES / name).read_bytes(), name)


def test_visible_text_is_not_read_twice_and_blocks_merge_by_top():
    """보이는 글자(쪽 위 줄·아래 쪽 번호)도 렌더 그림에 그려져 OCR이 읽지만 텍스트 레이어가 우선이라 버린다.
    블록은 윗변 순서: 위 텍스트 레이어 → OCR 문단 → 아래 쪽 번호."""
    pytest.importorskip("onnxruntime")
    require_models(*ocr.MODEL_NAMES)
    visible = [(40, 370, 12, "보이는 글자 줄"), (148, 20, 9, "1")]
    data = pdf(lambda c: scanned_page(c, 300, 400, BODY, visible))
    assert blocks(data) == [("paragraph", "text_layer", "보이는 글자 줄"),
                            ("paragraph", "ocr", "스캔한 쪽의 글자를 읽는다.\n두 줄로 된 문단이다."),
                            ("paragraph", "ocr", "다음 문단은 한 줄이다."),
                            ("paragraph", "text_layer", "1")]


def test_ocr_blocks_follow_the_previous_heading():
    pytest.importorskip("onnxruntime")
    require_models(*ocr.MODEL_NAMES)
    def digital(c):
        put(c, 40, 340, 18, "1. 추진 배경")
        for i in range(4):
            put(c, 40, 300 - 16 * i, 11, "본문 크기를 정하는 디지털 쪽의 글자다.")

    parsed = PdfParser().parse(pdf(digital, lambda c: scanned_page(c, 300, 400, BODY[2:])), "s.pdf")
    ocr_blocks = [b for b in parsed.blocks if b["text_source"] == "ocr"]
    assert [(b["text"], b["section_path"]) for b in ocr_blocks] == [("다음 문단은 한 줄이다.", ("1. 추진 배경",))]


def test_rotated_scanned_page_has_visible_page_coordinates():
    """/Rotate 90 쪽: 그림을 PDF 좌표에서 돌려 넣어 보이는 쪽에서 바로 선다. 결과는 돌리지 않은 쪽과 같다."""
    pytest.importorskip("onnxruntime")
    require_models(*ocr.MODEL_NAMES)
    image = text_image(300, 400, BODY)

    def upright(c):
        c.drawImage(ImageReader(image), 0, 0, width=300, height=400)

    def rotated(c):
        c.setPageRotation(90)  # reportlab은 MediaBox를 400×300으로 눕힌다. PDF 점 (x, y)는 보이는 (y, x)
        c.translate(400, 0)
        c.rotate(90)
        c.drawImage(ImageReader(image), 0, 0, width=300, height=400)

    plain = PdfParser().parse(pdf(upright), "u.pdf")
    turned = PdfParser().parse(pdf(rotated), "r.pdf")
    assert turned.pages[0].rotation == 90 and (turned.pages[0].width_pt, turned.pages[0].height_pt) == (300, 400)
    assert [b["text"] for b in turned.blocks] == [b["text"] for b in plain.blocks] and len(plain.blocks) == 2
    for a, b in zip(plain.blocks, turned.blocks):
        assert all(abs(a["locator"]["bbox"][k] - b["locator"]["bbox"][k]) <= 0.01 for k in ("x0", "y0", "x1", "y1"))


def test_blank_scanned_page_has_no_ocr_blocks():
    data = pdf(lambda c: scanned_page(c, 300, 400, [], visible=[(148, 20, 9, "1")]))
    assert blocks(data) == [("paragraph", "text_layer", "1")]


@pytest.mark.parametrize("name", ["image_page.pdf", "scanned_invisible.pdf"])
def test_scanned_golden_inputs_are_the_same_with_ocr_on(name):
    """기존 scanned 골든 예제(보이는 쪽 번호만, 숨은 글자층): OCR을 켜도 쪽 번호를 두 번 내지 않는다."""
    pytest.importorskip("onnxruntime")
    require_models(*ocr.MODEL_NAMES)
    data = (FIXTURES / name).read_bytes()
    assert PdfParser(ocr=True).parse(data, name) == PdfParser(ocr=False).parse(data, name)


def test_parallel_parses_give_the_same_ocr_blocks():
    """여러 스레드가 동시에 스캔 쪽을 파싱해도(렌더는 PDFIUM_LOCK, OCR 세션은 공유) 결과가 같다."""
    pytest.importorskip("onnxruntime")
    require_models(*ocr.MODEL_NAMES)
    from concurrent.futures import ThreadPoolExecutor

    data = pdf(lambda c: scanned_page(c, 300, 400, BODY))
    expected = PdfParser().parse(data, "s.pdf")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: PdfParser().parse(data, "s.pdf"), range(8)))
    assert all(r == expected for r in results) and len(expected.blocks) == 2
