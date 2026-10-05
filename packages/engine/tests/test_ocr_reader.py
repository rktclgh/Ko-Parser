"""OCR 실행부: 합성 그림(번들 Noto Sans KR로 그린 깨끗한 한글 줄)을 정확히 읽는지, 세션은 한 번만 만드는지,
OCR 추가 설치가 없으면 OcrUnavailable인지. 그림은 OS 글꼴에 기대지 않으려고 ko-parser-fonts 글꼴로 그린다."""

import builtins
import shutil
import subprocess
import sys
import threading
import time
import tracemalloc
import unicodedata
from types import SimpleNamespace

import pytest

ort = pytest.importorskip("onnxruntime")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ko_parser_fonts
import ko_parser_ocr_models
from ko_parser.errors import KoParserError, OcrUnavailable
from ko_parser.formats.pdf import ocr
from ko_parser.formats.pdf.ocr import reader, rec

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
    """아주 길쭉한 그림(3000×20)은 긴 변 2000px로 줄여(1984×32) 검출에 넘기고, 검출 모델 입력은 짧은 변을 키우되
    긴 변 4000px 상한 안이다(시간으로 보지 않는다)."""
    session = ocr.get_reader().det
    real = session.run
    seen = []

    def spy(names, feed):
        seen.append(next(iter(feed.values())).shape)
        return real(names, feed)

    monkeypatch.setattr(session, "run", spy)
    assert ocr.read_lines(Image.new("RGB", (3000, 20), "white")) == []
    assert seen == [(1, 3, 64, 4000)]


def test_zero_size_image_has_no_lines():
    assert ocr.read_lines(Image.new("RGB", (0, 0))) == []
    assert ocr.read_lines(Image.new("RGB", (3000, 0))) == []
    assert ocr.read_lines(Image.new("RGB", (5000, 0))) == []  # 미리 줄이는 경로(긴 변 > 4000)에서도
    assert ocr.read_lines(Image.new("RGB", (0, 5000))) == []


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
    assert ocr._reader is None
    assert issubclass(OcrUnavailable, KoParserError)


def test_broken_native_install_is_unavailable(monkeypatch):
    """import가 ImportError 밖의 오류(깨진 공유 라이브러리 등)를 내도 available()은 False, get_reader()는 OcrUnavailable."""
    real = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "onnxruntime":
            raise OSError("cannot load shared library")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    monkeypatch.setattr(ocr, "_reader", None)
    assert ocr.available() is False
    with pytest.raises(OcrUnavailable, match=r"missing onnxruntime"):
        ocr.get_reader()


def test_missing_model_file_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(ko_parser_ocr_models, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(ocr, "_reader", None)
    assert ocr.available() is False
    with pytest.raises(OcrUnavailable, match="det.onnx"):
        ocr.get_reader()
    assert ocr._reader is None


def test_broken_model_file_is_unavailable(monkeypatch, tmp_path):
    for name in (ko_parser_ocr_models.DET_FILE, ko_parser_ocr_models.REC_FILE, ko_parser_ocr_models.DICT_FILE):
        (tmp_path / name).write_bytes(b"not a model")
    monkeypatch.setattr(ko_parser_ocr_models, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(ocr, "_reader", None)
    assert ocr.available() is True  # 파일은 있다
    with pytest.raises(OcrUnavailable, match=r"could not be loaded: .*det\.onnx.* or run with --no-ocr") as info:
        ocr.get_reader()
    assert info.value.__cause__ is not None and ocr._reader is None


def test_dict_that_does_not_match_the_model_is_unavailable(monkeypatch, tmp_path):
    """사전 글자 수가 인식 모델의 출력 갈래 수와 다르면 글자 번호가 어긋난다: 만들 때 알린다."""
    root = ko_parser_ocr_models.model_dir()
    for name in (ko_parser_ocr_models.DET_FILE, ko_parser_ocr_models.REC_FILE):
        shutil.copyfile(root / name, tmp_path / name)
    words = (root / ko_parser_ocr_models.DICT_FILE).read_text(encoding="utf-8").removesuffix("\n").split("\n")
    (tmp_path / ko_parser_ocr_models.DICT_FILE).write_text("\n".join(words[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dict.txt"):
        reader.OcrReader(tmp_path, ko_parser_ocr_models.DET_FILE, ko_parser_ocr_models.REC_FILE,
                         ko_parser_ocr_models.DICT_FILE)
    monkeypatch.setattr(ko_parser_ocr_models, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(ocr, "_reader", None)
    with pytest.raises(OcrUnavailable, match=r"could not be loaded: .*dict\.txt"):
        ocr.get_reader()
    assert ocr._reader is None


def test_out_of_memory_is_not_reported_as_a_broken_install(monkeypatch):
    def oom(*args):
        raise MemoryError

    monkeypatch.setattr(reader, "OcrReader", oom)
    monkeypatch.setattr(ocr, "_reader", None)
    with pytest.raises(MemoryError):
        ocr.get_reader()
    assert ocr._reader is None


def test_ocr_module_imports_without_numpy():
    """엔진 기본 설치(numpy·onnxruntime 없음)에서도 OCR 모듈 import는 된다(실행부는 get_reader가 가져온다).
    이 프로세스의 모듈을 다시 읽지 않으려고 하위 프로세스에서 본다."""
    code = ("import sys; sys.modules['numpy'] = None; sys.modules['onnxruntime'] = None\n"
            "from ko_parser.formats.pdf import ocr\n"
            "print(ocr.available())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_release_is_registered_at_exit():
    """하위 프로세스: ocr보다 먼저 등록한 atexit 함수는 나중에 돈다(등록 역순). 그때 세션은 이미 놓여 있다."""
    code = ("import atexit, sys\n"
            "atexit.register(lambda: print(sys.modules['ko_parser.formats.pdf.ocr']._reader is None))\n"
            "from ko_parser.formats.pdf import ocr\n"
            "ocr.get_reader()\n"
            "print(ocr._reader is None)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "True"]


def test_reader_runs_only_detection_and_recognition():
    """방향 판정(180° 분류)은 쓰지 않는다: 긴 한국어 줄을 뒤집어 글자를 잃는다(스펙 §2). 세션은 검출·인식 둘뿐."""
    got = ocr.get_reader()
    sessions = sorted(k for k, v in vars(got).items() if isinstance(v, ort.InferenceSession))
    assert sessions == ["det", "rec"]
    assert [s.get_inputs()[0].shape[1] for s in (got.det, got.rec)] == [3, 3]


def test_release_waits_for_a_reader_being_built():
    """_release는 잠금을 잡고 놓는다: 다른 스레드가 읽개를 만드는 중이면 끝날 때까지 기다린다."""
    done = threading.Event()
    with ocr._lock:
        t = threading.Thread(target=lambda: (ocr._release(), done.set()))
        t.start()
        assert not done.wait(0.2)  # 잠금을 쥐고 있는 동안은 끝나지 않는다
    t.join()
    assert done.is_set() and ocr._reader is None


class FakeRec:
    """출력 갈래 수가 정해지지 않은(symbolic) 인식 모델 흉내. 글자 목록은 모델 정보에만 있다."""

    def __init__(self, character):
        self.character = character

    def get_outputs(self):
        return [SimpleNamespace(shape=["batch", "time", "classes"])]

    def get_modelmeta(self):
        meta = {} if self.character is None else {"character": self.character}
        return SimpleNamespace(custom_metadata_map=meta)


@pytest.mark.parametrize(("character", "ok"), [("가\n나\n", True), ("가\n다\n", False), (None, True)])
def test_dict_is_checked_against_model_metadata_when_classes_are_symbolic(monkeypatch, tmp_path, character, ok):
    (tmp_path / "dict.txt").write_text("가\n나\n", encoding="utf-8")
    monkeypatch.setattr(reader, "_session", lambda path: FakeRec(character))
    if ok:
        assert reader.OcrReader(tmp_path, "det", "rec", "dict.txt").symbols == ["blank", "가", "나", " "]
    else:
        with pytest.raises(ValueError, match="dict.txt"):
            reader.OcrReader(tmp_path, "det", "rec", "dict.txt")


def test_out_of_memory_during_dependency_import_is_not_reported_as_missing(monkeypatch):
    """의존성 import 중 메모리 부족은 '설치 없음'으로 바꾸지 않고 그대로 낸다."""
    real = builtins.__import__

    def oom(name, *args, **kwargs):
        if name == "numpy":
            raise MemoryError
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", oom)
    monkeypatch.setattr(ocr, "_reader", None)
    with pytest.raises(MemoryError):
        ocr.available()
    with pytest.raises(MemoryError):
        ocr.get_reader()
    assert ocr._reader is None


class FakeRecSession:
    """인식 모델 대신: 입력 모양을 적어 두고 빈칸만 고른 확률을 돌려준다."""

    def __init__(self):
        self.shapes = []

    def get_inputs(self):
        return [SimpleNamespace(name="x")]

    def run(self, _, feed):
        x = feed["x"]
        self.shapes.append(x.shape)
        return [np.zeros((x.shape[0], x.shape[3] // 8, 3), np.float32)]


def test_recognition_batches_stay_within_the_width_budget():
    """작은 글자의 긴 줄 여섯(2000×66 그림에서 나오는 높이 11·너비 1990 줄)도 한 번에 넣는 묶음 수 × 너비가
    예산 안이다(출력 텐서가 수백 MB로 커지지 않게)."""
    session = FakeRecSession()
    assert len(rec.recognize(session, ["blank", "가", " "], [np.zeros((11, 1990, 3), np.uint8)] * 6)) == 6
    assert sum(n for n, *_ in session.shapes) == 6
    assert all(n * w <= rec.REC_WIDTH_BUDGET for n, _, _, w in session.shapes)


def test_short_lines_are_still_read_six_at_a_time():
    session = FakeRecSession()
    rec.recognize(session, ["blank", "가", " "], [np.zeros((40, 400, 3), np.uint8)] * 7)
    assert session.shapes == [(6, 3, 48, 480), (1, 3, 48, 480)]


def test_a_line_wider_than_the_budget_is_read_alone_at_full_width():
    """예산보다 넓은 줄 하나는 줄이지 않고 혼자 읽는다(줄이면 글자가 바뀐다)."""
    session = FakeRecSession()
    crops = [np.zeros((10, 5000, 3), np.uint8), np.zeros((40, 400, 3), np.uint8)]
    assert len(rec.recognize(session, ["blank", "가", " "], crops)) == 2
    assert session.shapes == [(1, 3, 48, 480), (1, 3, 48, 24000)]


def test_huge_image_is_shrunk_with_pillow_first_and_boxes_stay_in_input_pixels(monkeypatch):
    """긴 변이 PRE_SHRINK_SIDE(4000)를 넘는 그림은 numpy 배열로 바꾸기 전에 Pillow로 정수배 줄인다(원래 크기 배열을
    만들지 않는다). 상자는 원래 그림 화소로 되돌린다."""
    font = ImageFont.truetype(str(ko_parser_fonts.font_dir() / ko_parser_fonts.FONT_FILE), 220)
    img = Image.new("RGB", (20000, 300), "white")
    draw = ImageDraw.Draw(img)
    starts = (500, 7500, 14500)
    for x in starts:
        draw.text((x, 20), "사업 계획", font=font, fill="black")
    seen = []
    real = reader.resize_linear

    def spy(a, width, height):
        seen.append(a.shape)
        return real(a, width, height)

    monkeypatch.setattr(reader, "resize_linear", spy)
    tracemalloc.start()
    try:
        lines = ocr.read_lines(img)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert seen == [(60, 4000, 3)]  # 5배 줄인 그림(4000×60)을 다시 긴 변 2000으로
    assert peak < 200 * 2**20, peak
    assert [ln.text for ln in lines] == ["사업 계획"] * 3
    for ln, x in zip(lines, starts):
        assert abs(left_top(ln)[0] - x) <= 80
        assert all(0 <= px <= 20000 and 0 <= py <= 300 for px, py in ln.box)


def test_pre_shrink_keeps_normal_pages_on_the_same_path(monkeypatch):
    """보통 쪽(긴 변 4000 이하, 1600만 화소 이하)은 예전 경로 그대로(상한을 없앤 것과 바이트까지 같다). 상한을 낮춰 미리 줄이게 해도
    글자는 같고 상자는 원래 그림 화소(몇 화소 차)."""
    image = page_image(scale=2)  # 2000×800
    default = ocr.read_lines(image)
    monkeypatch.setattr(reader, "PRE_SHRINK_SIDE", 10**9)
    monkeypatch.setattr(reader, "PRE_SHRINK_PIXELS", 10**18)
    assert ocr.read_lines(image) == default
    monkeypatch.setattr(reader, "PRE_SHRINK_SIDE", 1000)  # 2배 줄여 1000×400으로 읽는다
    shrunk = ocr.read_lines(image)
    assert [ln.text for ln in shrunk] == [ln.text for ln in default] == LINES
    for a, b in zip(default, shrunk):
        assert all(abs(pa - pb) <= 12 for p, q in zip(a.box, b.box) for pa, pb in zip(p, q))


def test_large_square_image_keeps_memory_bounded():
    """8000×8000(6400만 화소)도 먼저 2배 줄여 4000×4000으로 읽는다: numpy 쪽 최대 할당이 250MB 아래."""
    img = Image.new("RGB", (8000, 8000), "white")
    tracemalloc.start()
    try:
        assert ocr.read_lines(img) == []
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 250 * 2**20, peak
