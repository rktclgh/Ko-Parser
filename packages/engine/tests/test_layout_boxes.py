"""레이아웃 실행부: 스파이크와 같은 전처리·NMS·후처리(합성 배열), 실제 모델은 onnxruntime이 있을 때만."""

import io
import os
import threading
from pathlib import Path

import pytest
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from hanji import models
from hanji.formats.pdf import layout, scan
from hanji.formats.pdf.layout import LayoutBox

np = pytest.importorskip("numpy")
from hanji.formats.pdf.layout import boxes  # noqa: E402  numpy가 있어야 import된다

FONT = "HYGothic-Medium"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))
PX = 200 / 72  # 렌더 화소 / pt
CAPTION = "그림 1. 분기별 처리 건수(단위: 건)"
# 막대 차트를 그린 자리(보이는 쪽 pt, 축 이름표 포함). 2026-10-06 macOS 실측 모델 상자 (91.3, 127.7, 490.9, 336.9), IoU 0.978
CHART = (94.0, 128.0, 491.0, 334.0)


def put(c: Canvas, x: float, y: float, size: float, s: str, font: str = FONT) -> None:
    c.setFont(font, size)
    c.drawString(x, y, s)


def bar_chart_page() -> bytes:
    """막대 차트(선·도형, 이름표 Helvetica) + 아래 캡션 + 문단. 2026-10-06 macOS 실측(공식 ONNX): chart 0.933,
    figure_title 0.691."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(595.0, 842.0), invariant=1, pageCompression=0)
    put(c, 60, 790, 16, "합성 보고서: 차트 예제")
    put(c, 60, 760, 10.5, "이 쪽은 레이아웃 실행부 시험을 위한 지어낸 글이다.")
    x0, y0, w, h = 110, 520, 380, 190
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
    put(c, 200, 480, 10, CAPTION)
    put(c, 60, 440, 10.5, "차트 아래 문단은 그대로 문단이다.")
    c.showPage()
    c.save()
    return buf.getvalue()


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def pt(box) -> tuple[float, ...]:
    return tuple(v / PX for v in box)


def require_models(*names: str) -> None:
    """실제 모델 파일이 필요한 테스트: 찾을 수 없으면 건너뛰고, HANJI_CI_REQUIRE_MODELS=1이면 실패한다."""
    missing = [name for name in names if models.find(name) is None]
    if missing:
        if os.environ.get("HANJI_CI_REQUIRE_MODELS") == "1":
            pytest.fail(f"model files not found: {missing}; run `hanji models fetch`")
        pytest.skip(f"model files not found: {missing} (hanji models fetch)")


def test_preprocess_is_800_bicubic_rgb_scaled_to_0_1():
    image = Image.new("RGB", (400, 200), (255, 0, 0))
    image.paste((0, 0, 255), (200, 0, 400, 200))  # 오른쪽 절반 파랑
    feed = boxes.preprocess(image)
    x = feed["image"]
    assert x.shape == (1, 3, 800, 800) and x.dtype == np.float32
    assert (x[0, 0, 400, 100], x[0, 2, 400, 100]) == (1.0, 0.0)  # 왼쪽 빨강: 채널 순서는 RGB
    assert (x[0, 0, 400, 700], x[0, 2, 400, 700]) == (0.0, 1.0)
    assert feed["im_shape"].tolist() == [[800.0, 800.0]] and feed["scale_factor"].tolist() == [[4.0, 2.0]]
    expected = np.asarray(image.resize((800, 800), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    assert np.array_equal(x[0], expected.transpose(2, 0, 1))
    assert np.allclose(boxes.preprocess(Image.new("L", (10, 10), 51))["image"], 0.2)  # 흑백도 RGB로


def test_preprocess_does_not_copy_an_rgb_image(monkeypatch):
    """RGB 쪽 그림은 convert로 통째 복사하지 않는다(렌더는 한 변 4000까지, 약 48MB). 다른 모드만 RGB로 바꾼다."""
    converted = []
    convert = Image.Image.convert

    def spy(self, *args, **kwargs):
        converted.append(self.mode)
        return convert(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", spy)
    boxes.preprocess(Image.new("RGB", (40, 20)))
    boxes.preprocess(Image.new("L", (40, 20)))
    assert converted == ["L"]


def test_nms_suppresses_same_class_above_06_and_other_classes_above_098():
    a = LayoutBox("chart", 0.9, (0, 0, 100, 100))
    over = LayoutBox("chart", 0.8, (0, 0, 100, 160))  # 같은 분류 IoU 0.625 > 0.6: 버림
    under = LayoutBox("chart", 0.7, (0, 50, 100, 220))  # IoU 0.227: 남김
    assert boxes.nms([under, over, a]) == [a, under]
    other_kept = LayoutBox("image", 0.6, (0, 0, 100, 103))  # 다른 분류 IoU 0.971 ≤ 0.98: 남김
    other_dropped = LayoutBox("table", 0.5, (0, 0, 100, 101))  # 0.990 > 0.98: 버림
    assert boxes.nms([a, other_kept, other_dropped]) == [a, other_kept]


def test_postprocess_drops_low_unknown_nonfinite_and_empty_and_clips():
    rows = np.array([
        [16, 0.15, 0, 0, 100, 100],  # 바닥 0.2 미만
        [16, 0.50, -10, 5, 120, 90],  # 그림 밖은 자른다
        [1, 0.30, 10, 10, 10, 50],  # 넓이 0
        [-1, 0.99, 0, 0, 5, 5],  # 빈 칸
        [99, 0.99, 0, 0, 5, 5],  # 모르는 분류
        [6, np.nan, 0, 0, 5, 5],
    ], dtype=np.float32)
    expected = [LayoutBox("chart", 0.5, (0.0, 5.0, 100.0, 80.0))]
    assert boxes.postprocess(rows, 100, 80) == expected
    assert boxes.postprocess(np.hstack([rows, np.zeros((6, 1), np.float32)]), 100, 80) == expected  # 순서 열이 있어도
    assert boxes.postprocess(np.zeros((0, 6), np.float32), 100, 80) == []
    with pytest.raises(ValueError, match="2-D"):  # 배치 축이 붙은 (1, N, 6)을 한 행으로 펴 상자를 잃지 않는다
        boxes.postprocess(rows[None], 100, 80)
    assert boxes.LABELS[1] == "image" and boxes.LABELS[6] == "figure_title" and boxes.LABELS[16] == "chart"


def test_read_labels_takes_the_label_list_block(tmp_path):
    config = tmp_path / "inference.yml"
    config.write_text("mode: paddle\nlabel_list:\n- paragraph_title\n- image\nHpi:\n  x:\n  - 1\n", encoding="utf-8")
    assert boxes.read_labels(config) == ("paragraph_title", "image")
    config.write_text("label_list:\n- paragraph_title\n\n- image\nHpi: 1\n", encoding="utf-8")  # 블록 안 빈 줄
    assert boxes.read_labels(config) == ("paragraph_title", "image")
    config.write_text("mode: paddle\n", encoding="utf-8")
    assert boxes.read_labels(config) == ()


def test_official_inference_yml_lists_the_same_labels():
    """모델 파일과 함께 받는 설정의 분류 목록(20개, 순서 그대로)이 엔진이 쓰는 LABELS와 같다."""
    require_models("layout-config")
    assert boxes.read_labels(models.resolve("layout-config")) == boxes.LABELS and len(boxes.LABELS) == 20


def test_detector_finds_the_bar_chart_and_its_caption():
    pytest.importorskip("onnxruntime")
    require_models(*layout.MODEL_NAMES)
    image = scan.render(bar_chart_page(), "chart.pdf", 0)
    found = layout.detect(image)
    charts = [b for b in found if b.cls == "chart" and b.score >= 0.5]
    captions = [b for b in found if b.cls == "figure_title" and b.score >= 0.3]
    caption_truth = (200.0, 354.5, 200.0 + pdfmetrics.stringWidth(CAPTION, FONT, 10), 363.4)
    assert len(charts) == 1 and iou(pt(charts[0].box), CHART) >= 0.8
    assert any(iou(pt(b.box), caption_truth) >= 0.5 for b in captions)
    assert all(0 <= b.box[0] < b.box[2] <= image.width and 0 <= b.box[1] < b.box[3] <= image.height for b in found)
    assert [b.score for b in found] == sorted((b.score for b in found), reverse=True)


def test_blank_and_tiny_images_are_safe():
    pytest.importorskip("onnxruntime")
    require_models(*layout.MODEL_NAMES)
    blank = layout.detect(Image.new("RGB", (1653, 2339), "white"))
    assert not [b for b in blank if b.cls in ("image", "chart") and b.score >= 0.4]
    tiny = layout.detect(Image.new("L", (1, 1), 0))
    assert all(0 <= b.box[0] < b.box[2] <= 1 and 0 <= b.box[1] < b.box[3] <= 1 for b in tiny)


def test_onnx_file_of_another_model_or_other_labels_are_rejected(tmp_path):
    """입력 이름이 PP-DocLayout과 다른 ONNX(여기서는 OCR 검출 모델)나 분류 목록이 다른 설정은 검출기가 ValueError로
    거절한다(_build가 LayoutUnavailable로 바꾼다). 고정한 SHA-256과 다른 파일은 그 전에 resolve가 거절한다
    (test_layout_runtime). 분류 목록은 세션을 열기 전에 본다."""
    pytest.importorskip("onnxruntime")
    require_models("ocr-det", "layout-config")
    from hanji.formats.pdf.layout.detector import LayoutDetector

    with pytest.raises(ValueError, match=r"is not a PP-DocLayout model \(inputs \["):
        LayoutDetector(models.resolve("ocr-det"), models.resolve("layout-config"))
    other = tmp_path / "inference.yml"
    other.write_text("label_list:\n- text\n- image\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"lists 2 labels that are not the PP-DocLayout_plus-L labels"):
        LayoutDetector(Path("no-such-model.onnx"), other)


def test_concurrent_detection_gives_the_same_boxes():
    pytest.importorskip("onnxruntime")
    require_models(*layout.MODEL_NAMES)
    image = scan.render(bar_chart_page(), "chart.pdf", 0)
    expected = layout.detect(image)
    results = []
    threads = [threading.Thread(target=lambda: results.append(layout.detect(image))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert expected and results == [expected] * 4
