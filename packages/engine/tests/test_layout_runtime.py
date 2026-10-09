"""레이아웃 실행부의 설치 확인·세션 캐시. onnxruntime·실제 모델 없이 돈다(설치 확인과 만들기를 바꿔 끼우고, 모델은
HANJI_MODEL_DIR의 가짜 파일)."""

import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from hanji import models
from hanji.errors import LayoutUnavailable
from hanji.formats.pdf import layout


def fake_model(monkeypatch, tmp_path) -> None:
    """찾기만 되는 가짜 레이아웃 모델·설정 파일(HANJI_MODEL_DIR, 해시는 틀리다)."""
    for name in layout.MODEL_NAMES:
        path = tmp_path / "models" / models.MANIFEST[name].path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a model")
    monkeypatch.setenv("HANJI_MODEL_DIR", str(tmp_path / "models"))


def broken_module(monkeypatch, tmp_path, name: str = "onnxruntime") -> None:
    """설치는 됐는데(찾을 수 있다) import하면 깨지는 모듈을 sys.path 맨 앞에 둔다. 진짜 모듈은 잠깐 뺀다. 모델 파일은
    찾히게 둔다(가짜)."""
    root = tmp_path / "broken"
    (root / name).mkdir(parents=True)
    (root / name / "__init__.py").write_text(
        'raise ImportError("libstub.so.1: cannot open shared object file")\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(layout, "_detector", None)
    fake_model(monkeypatch, tmp_path)


def test_available_does_not_import_the_runtime():
    """설치 확인은 numpy·onnxruntime을 import하지 않는다(하위 프로세스에서 본다)."""
    code = ("import sys\n"
            "from hanji.formats.pdf import layout\n"
            "layout.available()\n"
            "print(sorted(m for m in ('numpy', 'onnxruntime') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_missing_install_is_unavailable_with_the_install_hint(monkeypatch):
    monkeypatch.setattr(layout, "MODULES", ("no_such_layout_module", *layout.MODULES))
    monkeypatch.setattr(layout, "_detector", None)
    assert layout.available() is False
    with pytest.raises(LayoutUnavailable,
                       match=r"not installed \(missing no_such_layout_module\).*hanji\[layout\].*--no-layout"):
        layout.get_detector()


def test_installed_but_broken_module_is_available_and_get_detector_names_it(monkeypatch, tmp_path):
    """available()은 '설치됐는가'(찾기만): 깨진 설치도 참이고 get_detector()가 모듈과 오류를 담은 LayoutUnavailable."""
    pytest.importorskip("numpy")
    broken_module(monkeypatch, tmp_path)
    assert layout.available() is True
    with pytest.raises(LayoutUnavailable,
                       match=r"onnxruntime .*libstub\.so\.1.*hanji\[layout\].* or run with --no-layout"):
        layout.get_detector()
    assert layout._detector is None


def test_missing_model_file_means_not_installed(monkeypatch):
    """모델 파일을 찾을 수 없으면(fetch 전) 설치가 없는 것과 같다(리뷰 C2): available 거짓(자동 모드는 조용히 끈다),
    get_detector()는 fetch 안내를 담은 LayoutUnavailable."""
    monkeypatch.setattr(models, "find", lambda name: None)
    monkeypatch.setattr(layout, "MODULES", ())
    monkeypatch.setattr(layout, "_detector", None)
    assert layout.available() is False
    with pytest.raises(LayoutUnavailable, match=r"layout model file layout/inference\.onnx not found; run `hanji "
                                                r"models fetch layout` or set HANJI_MODEL_DIR, or run with --no-layout"):
        layout.get_detector()
    assert layout._detector is None


def test_wrong_model_file_is_a_broken_install(monkeypatch, tmp_path):
    """찾은 모델 파일이 고정한 SHA-256과 다르면 깨진 설치: available 참, get_detector()가 크게 알린다(세션까지 가지
    않는다)."""
    fake_model(monkeypatch, tmp_path)
    monkeypatch.setattr(layout, "MODULES", ())
    monkeypatch.setattr(layout, "_detector", None)
    assert layout.available() is True
    with pytest.raises(LayoutUnavailable, match=r"does not match the pinned size and SHA-256 of layout/inference\.onnx"
                                                r".*, or run with --no-layout") as info:
        layout.get_detector()
    assert isinstance(info.value.__cause__, models.ModelError) and layout._detector is None


def test_detector_is_built_once_across_threads(monkeypatch):
    built = []

    def slow_build():
        time.sleep(0.05)
        built.append(1)
        return object()

    monkeypatch.setattr(layout, "_detector", None)
    monkeypatch.setattr(layout, "_build", slow_build)
    got = []
    threads = [threading.Thread(target=lambda: got.append(layout.get_detector())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1 and len(got) == 8 and all(g is got[0] for g in got)


def test_release_drops_the_detector_and_rebuilds_on_demand(monkeypatch):
    """atexit로 등록한 _release가 세션을 놓는다(프로세스 끝 abort 방지). 놓은 뒤 다시 부르면 새로 만든다."""
    monkeypatch.setattr(layout, "_detector", None)
    monkeypatch.setattr(layout, "_build", object)
    first = layout.get_detector()
    layout._release()
    assert layout._detector is None and layout.get_detector() is not first


@pytest.mark.parametrize("wrong", ["labels", "inputs"])
def test_detector_that_rejects_the_model_is_unavailable_and_retried(monkeypatch, tmp_path, wrong):
    """LayoutDetector가 만들 때 낸 ValueError(분류 목록·입력 이름이 다르다)는 get_detector()에서 LayoutUnavailable이다.
    고정한 파일과 이 빌드가 맞지 않는 것이라 추가 설치를 다시 깔라고 하지 않는다(OCR 글자 목록 불일치와 같은 문구).
    실패는 붙잡아 두지 않는다: 다음 호출은 다시 만든다."""
    pytest.importorskip("onnxruntime")
    from hanji.formats.pdf.layout import detector

    config = tmp_path / "inference.yml"
    labels = ("text", "image") if wrong == "labels" else detector.LABELS
    config.write_text("label_list:\n" + "".join(f"- {name}\n" for name in labels), encoding="utf-8")
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="x")])  # 입력 이름이 다른 모델
    monkeypatch.setattr(detector.ort, "InferenceSession", lambda *args, **kwargs: session)
    monkeypatch.setattr(layout, "MODULES", ())
    monkeypatch.setattr(models, "find", lambda name: tmp_path / name)
    monkeypatch.setattr(models, "resolve", lambda name: config if name == "layout-config" else tmp_path / "x.onnx")
    monkeypatch.setattr(layout, "_detector", None)
    reason = "lists 2 labels that are not" if wrong == "labels" else r"is not a PP-DocLayout model \(inputs \['x'\]\)"
    with pytest.raises(LayoutUnavailable, match=rf"layout model could not be loaded: .*{reason}.*; "
                                                r"reinstall hanji or report it, or run with --no-layout$") as info:
        layout.get_detector()
    assert isinstance(info.value.__cause__, ValueError) and "[layout]" not in str(info.value)
    assert layout._detector is None
    monkeypatch.setattr(detector, "LayoutDetector", lambda model, config: "built")
    assert layout.get_detector() == "built"


@pytest.mark.parametrize(("preset", "expected"), [(None, "1"), ("0", "0")])
def test_layout_module_turns_off_onnxruntime_telemetry_before_import(preset, expected):
    """OCR 모듈과 같다: 레이아웃 모듈 import가 onnxruntime 원격 측정을 기본값으로 끈다(사용자가 정한 값은 둔다)."""
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    if preset is not None:
        env["ORT_DISABLE_TELEMETRY"] = preset
    code = "import os\nfrom hanji.formats.pdf import layout\nprint(os.environ['ORT_DISABLE_TELEMETRY'])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", check=False,
                         env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == expected
