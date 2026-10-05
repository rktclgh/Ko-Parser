"""OCR 화소 연산이 OpenCV(RapidOCR가 쓰는 연산)와 같은 값을 내는지. 기대값은 opencv-python-headless 4.x로 한 번 잰 값."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pyclipper")

from ko_parser.formats.pdf.ocr.det import _input_size, components, fill_poly, mini_box
from ko_parser.formats.pdf.ocr.pixels import crop_quad, resize_linear


def gray3(rows):
    return np.array(rows, np.uint8)[:, :, None].repeat(3, 2)


def test_resize_linear_matches_opencv_up_and_down():
    up = resize_linear(gray3([[0, 64, 255], [128, 32, 16]]), 5, 3)
    assert up[:, :, 0].tolist() == [[0, 26, 64, 178, 255], [64, 58, 48, 100, 136], [128, 89, 32, 22, 16]]
    big = gray3((np.arange(7 * 9).reshape(7, 9) * 4 % 256).tolist())
    assert resize_linear(big, 4, 3)[:, :, 0].tolist() == [[26, 35, 44, 53], [111, 120, 129, 138], [194, 203, 212, 221]]


def test_resize_linear_same_size_is_the_same_array():
    img = gray3([[1, 2], [3, 4]])
    assert resize_linear(img, 2, 2) is img


def test_fill_poly_matches_opencv():
    mask = fill_poly((8, 10), [[1, 1], [8, 2], [7, 6], [2, 5]])
    assert mask.astype(int).tolist() == [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


def test_mini_box_matches_opencv_min_area_rect_order():
    box, short = mini_box(np.array([[10, 10], [40, 20], [35, 35], [5, 25], [20, 20]], float))
    assert box.round(3).tolist() == [[10.0, 10.0], [40.0, 20.0], [35.0, 35.0], [5.0, 25.0]]
    assert round(short, 3) == 15.811


def test_components_are_8_connected():
    mask = np.zeros((6, 8), bool)
    mask[1, 1] = mask[2, 2] = True  # 대각선으로만 닿는다: 한 덩어리
    mask[4, 5:8] = True
    groups = sorted(sorted(map(tuple, g.tolist())) for g in components(mask))
    assert groups == [[(1, 1), (1, 1), (2, 2), (2, 2)], [(5, 4), (7, 4)]]


def test_crop_quad_of_an_axis_aligned_box_is_the_sub_image():
    img = gray3((np.arange(20 * 30).reshape(20, 30) % 251).tolist())
    out = crop_quad(img, np.array([[3, 4], [23, 4], [23, 12], [3, 12]], np.float32))
    assert out.shape == (8, 20, 3) and (out == img[4:12, 3:23]).all()


def test_crop_quad_turns_a_tall_box_upright():
    img = gray3((np.arange(40 * 10).reshape(40, 10) % 251).tolist())
    out = crop_quad(img, np.array([[2, 2], [8, 2], [8, 32], [2, 32]], np.float32))
    assert out.shape == (6, 30, 3)  # 30×6 → 90° 돌려 6×30


def test_detector_input_is_capped_for_a_very_long_strip():
    """짧은 변을 736으로 키우되 긴 변은 4000을 넘지 않는다(아주 길쭉한 그림의 메모리). 보통 쪽은 그대로."""
    assert _input_size(1414, 2000) == (1408, 1984)  # 32 배수(파이썬 round: 62.5 → 62)
    assert _input_size(20, 2000) == (32, 4000)
    assert _input_size(10, 10) == (736, 736)
