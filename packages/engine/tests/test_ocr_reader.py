"""OCR 실행부: 합성 그림(번들 Noto Sans KR로 그린 깨끗한 한글 줄)을 정확히 읽는지, 세션은 한 번만 만드는지,
OCR 추가 설치가 없으면 OcrUnavailable인지. 그림은 OS 글꼴에 기대지 않으려고 ko-parser-fonts 글꼴로 그린다."""

import subprocess
import sys
import threading
import time
import unicodedata

import pytest

ort = pytest.importorskip("onnxruntime")

from PIL import Image, ImageDraw, ImageFont

import ko_parser_fonts
import ko_parser_ocr_models
from ko_parser.errors import KoParserError, OcrUnavailable
from ko_parser.formats.pdf import ocr
from ko_parser.formats.pdf.ocr import det, reader

LINES = ["스캔한 쪽의 글자를 읽는다.", "공공누리 2026년 10월 5일", "사업 계획 보고서"]


def page_image(lines=LINES, scale: int = 1, mode: str = "RGB") -> Image.Image:
    font = ImageFont.truetype(str(ko_parser_fonts.font_dir() / ko_parser_fonts.FONT_FILE), 40 * scale)
    img = Image.new(mode, (1000 * scale, 100 * (len(lines) + 1) * scale), "white")
    draw = ImageDraw.Draw(img)
    for i, s in enumerate(lines):
        draw.text((60 * scale, (60 + 100 * i) * scale), s, font=font, fill="black")
    return img


def left_top(line: ocr.OcrLine) -> tuple[float, float]:
    return min(x for x, _ in line.box), min(y for _, y in line.box)


def test_reads_clean_korean_lines_exactly_in_reading_order():
    lines = ocr.read_lines(page_image())
    assert [ln.text for ln in lines] == LINES
    assert all(ln.text == unicodedata.normalize("NFC", ln.text) and 0.9 <= ln.score <= 1 for ln in lines)
    for i, ln in enumerate(lines):  # 그린 자리(왼쪽 60px, 윗변 60 + 100i px 근처의 글자 윗부분)
        x, y = left_top(ln)
        assert len(ln.box) == 4 and abs(x - 60) <= 10 and abs(y - (65 + 100 * i)) <= 12


def test_grayscale_input_reads_the_same():
    assert [ln.text for ln in ocr.read_lines(page_image(mode="L"))] == LINES


def test_large_image_is_shrunk_but_boxes_stay_in_input_pixels():
    """긴 변이 2000px를 넘으면 줄여서 읽고 상자는 원래 그림 화소로 되돌린다."""
    small, big = ocr.read_lines(page_image()), ocr.read_lines(page_image(scale=3))
    assert [ln.text for ln in big] == LINES
    for a, b in zip(small, big):  # 줄인 그림에서 찾은 상자라 몇 화소 다르다(실측 최대 19px, 그림 너비 3000px)
        assert all(abs(3 * pa - pb) <= 30 for p, q in zip(a.box, b.box) for pa, pb in zip(p, q))


def test_blank_image_has_no_lines():
    assert ocr.read_lines(Image.new("RGB", (800, 600), "white")) == []


def test_very_long_strip_is_read_without_huge_memory(monkeypatch):
    """아주 길쭉한 그림은 긴 변 2000px로 줄여 검출에 넘기고, 검출 입력도 긴 변 4000px 상한 안이다(시간으로 보지 않는다)."""
    seen = []
    real = reader.detect

    def spy(session, img):
        seen.append(img.shape)
        return real(session, img)

    monkeypatch.setattr(reader, "detect", spy)
    assert ocr.read_lines(Image.new("RGB", (3000, 20), "white")) == []
    assert seen == [(32, 1984, 3)] and det._input_size(32, 1984) == (64, 4000)


def test_dict_is_the_recognition_model_alphabet():
    """사전 파일과 인식 모델 안의 글자 목록이 같아야 글자 번호가 맞는다."""
    root = ko_parser_ocr_models.model_dir()
    session = ort.InferenceSession(str(root / ko_parser_ocr_models.REC_FILE), providers=["CPUExecutionProvider"])
    inner = session.get_modelmeta().custom_metadata_map["character"].removesuffix("\n").split("\n")
    assert ocr.get_reader().symbols == ["blank", *inner, " "]


def test_reader_is_built_once_across_threads(monkeypatch):
    built = []

    def slow_build():
        time.sleep(0.05)
        built.append(1)
        return object()

    monkeypatch.setattr(ocr, "_reader", None)
    monkeypatch.setattr(ocr, "_build", slow_build)
    got = []
    threads = [threading.Thread(target=lambda: got.append(ocr.get_reader())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1 and len(got) == 8 and all(g is got[0] for g in got)


def test_sessions_are_released_at_exit_and_rebuilt_on_demand(monkeypatch):
    """atexit로 등록한 _release가 세션을 놓는다(프로세스 끝 abort 방지). 놓은 뒤 다시 부르면 새로 만든다."""
    monkeypatch.setattr(ocr, "_reader", None)
    first = ocr.get_reader()
    ocr._release()
    assert ocr._reader is None
    assert ocr.get_reader() is not first


def test_concurrent_reads_give_the_same_lines():
    image = page_image()
    expected = ocr.read_lines(image)
    results = []
    threads = [threading.Thread(target=lambda: results.append(ocr.read_lines(image))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [expected] * 4


@pytest.mark.parametrize("module", ["onnxruntime", "numpy", "pyclipper", "ko_parser_ocr_models"])
def test_missing_ocr_install_is_unavailable(monkeypatch, module):
    monkeypatch.setitem(sys.modules, module, None)  # import가 ImportError
    monkeypatch.setattr(ocr, "_reader", None)
    assert ocr.available() is False
    with pytest.raises(OcrUnavailable, match=r'missing .*ko-parser-engine\[ocr\].* or run with --no-ocr'):
        ocr.get_reader()
    assert issubclass(OcrUnavailable, KoParserError)


def test_missing_model_file_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(ko_parser_ocr_models, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(ocr, "_reader", None)
    assert ocr.available() is False
    with pytest.raises(OcrUnavailable, match="det.onnx"):
        ocr.get_reader()


def test_broken_model_file_is_unavailable(monkeypatch, tmp_path):
    for name in (ko_parser_ocr_models.DET_FILE, ko_parser_ocr_models.REC_FILE, ko_parser_ocr_models.DICT_FILE):
        (tmp_path / name).write_bytes(b"not a model")
    monkeypatch.setattr(ko_parser_ocr_models, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(ocr, "_reader", None)
    assert ocr.available() is True  # 파일은 있다
    with pytest.raises(OcrUnavailable, match=r"could not be loaded.* or run with --no-ocr"):
        ocr.get_reader()


def test_ocr_module_imports_without_numpy():
    """엔진 기본 설치(numpy·onnxruntime 없음)에서도 OCR 모듈 import는 된다(실행부는 get_reader가 가져온다).
    이 프로세스의 모듈을 다시 읽지 않으려고 하위 프로세스에서 본다."""
    code = ("import sys; sys.modules['numpy'] = None; sys.modules['onnxruntime'] = None\n"
            "from ko_parser.formats.pdf import ocr\n"
            "print(ocr.available())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_reader_runs_only_detection_and_recognition():
    """방향 판정(180° 분류)은 쓰지 않는다: 긴 한국어 줄을 뒤집어 글자를 잃는다(스펙 §2). 세션은 검출·인식 둘뿐."""
    reader = ocr.get_reader()
    assert sorted(vars(reader)) == ["det", "rec", "symbols"]
    assert [s.get_inputs()[0].shape[1] for s in (reader.det, reader.rec)] == [3, 3]
