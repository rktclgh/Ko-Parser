"""스캔 쪽 OCR 실행부(onnxruntime + numpy + Pillow + pyclipper, 모델은 ko-parser-ocr-models).

OCR 추가 설치(ko-parser-engine[ocr])가 없어도 이 모듈은 import된다: numpy·onnxruntime은 get_reader()가 처음 불릴
때 가져온다(available()은 찾기만 한다). 세션(검출·인식)은 프로세스에 하나이고 만들 때만 잠금으로 보호한다.
PDFium은 부르지 않는다."""

import atexit
import importlib.util
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ....errors import OcrUnavailable

if TYPE_CHECKING:
    from PIL import Image

    from .reader import OcrReader

__all__ = ["OcrLine", "OcrUnavailable", "available", "get_reader", "read_lines"]

INSTALL_HINT = 'pip install "ko-parser-engine[ocr]"'
MODULES = ("numpy", "onnxruntime", "pyclipper", "ko_parser_ocr_models")  # OCR 추가 설치가 까는 모듈


@dataclass(frozen=True, slots=True)
class OcrLine:
    """그림 속 줄 하나. box는 네 꼭짓점(그림 화소, 왼쪽 위부터 시계 방향), text는 NFC, score는 글자 확률 평균(0~1)."""

    box: tuple[tuple[float, float], ...]
    text: str
    score: float


_lock = threading.Lock()
_reader: "OcrReader | None" = None


def _installed(module: str) -> bool:
    """모듈을 찾을 수 있는가(import하지 않는다). 찾다가 난 오류(깨진 상위 패키지, __spec__ 없는 모듈)는 없는 것."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _missing() -> str | None:
    """OCR 추가 설치에서 없는 것(모듈 이름이나 모델 파일). 다 있으면 None. 모듈은 찾기만 하고 import하지 않는다:
    깔렸는데 import가 깨지는 설치는 '있음'이고 _build가 크게 알린다(자동 모드가 조용히 물러나지 않게)."""
    for module in MODULES:
        if not _installed(module):
            return module
    try:
        import ko_parser_ocr_models as models  # 순수 파이썬 작은 패키지(모델 파일 이름·폴더)
    except MemoryError:  # 메모리 부족은 설치 문제가 아니다
        raise
    except Exception:  # 깔렸는데 import가 깨진다: 모델 파일은 못 보지만 '있음'으로 두고 _build가 알린다
        return None
    for name in (models.DET_FILE, models.REC_FILE, models.DICT_FILE):
        if not (models.model_dir() / name).is_file():
            return f"ko_parser_ocr_models/{name}"
    return None


def available() -> bool:
    """OCR 추가 설치가 있는가(필요한 모듈을 찾을 수 있고 모델 파일이 있다). numpy·onnxruntime을 import하지 않고
    세션도 만들지 않는다. 깔렸는데 깨진 설치도 참이다(get_reader()가 OcrUnavailable)."""
    return _missing() is None


def _build() -> "OcrReader":
    missing = _missing()
    if missing is not None:
        raise OcrUnavailable(f"OCR is not installed (missing {missing}); {INSTALL_HINT} or run with --no-ocr")
    for module in MODULES:  # 깔렸는데 import가 깨지는 의존성(공유 라이브러리·glibc 등): 모듈과 오류를 알린다
        try:
            __import__(module)
        except MemoryError:  # 메모리 부족은 설치 문제가 아니다
            raise
        except Exception as exc:
            raise OcrUnavailable(
                f"OCR dependency {module} is installed but could not be imported "
                f"({type(exc).__name__}: {exc}); reinstall with {INSTALL_HINT} or run with --no-ocr"
            ) from exc
    import ko_parser_ocr_models as models

    from .reader import OcrReader

    root = models.model_dir()
    try:
        return OcrReader(root, models.DET_FILE, models.REC_FILE, models.DICT_FILE)
    except MemoryError:  # 메모리 부족은 설치 문제가 아니다
        raise
    except Exception as exc:  # 깨진 모델 파일·사전·onnxruntime 오류: 설정 문제로 알린다(오류에 파일 경로가 있다)
        raise OcrUnavailable(
            f"OCR models in {root} could not be loaded: {exc}; reinstall with {INSTALL_HINT} or run with --no-ocr"
        ) from exc


def get_reader() -> "OcrReader":
    """프로세스에 하나뿐인 읽개. 처음 부를 때 세션을 만든다(동시에 불러도 한 번만). 설치가 없거나 깨졌으면
    OcrUnavailable. 부르는 쪽은 읽개를 인터프리터 종료 너머까지 붙잡아 두지 않는다(atexit의 _release가 놓을 수 있게)."""
    global _reader
    if _reader is None:
        with _lock:
            if _reader is None:
                _reader = _build()
    return _reader


def _release() -> None:
    """프로세스가 끝날 때 세션을 먼저 놓는다. onnxruntime 작업 스레드가 C++ 정적 소멸자와 겹치면 macOS에서 드물게
    'recursive_mutex lock failed'로 abort한다(실측: 10쪽 읽고 끝나는 프로세스 약 60번에 1번).
    인터프리터가 끝날 때 아직 추론 중인 데몬 스레드는 지원하지 않는다(그 스레드가 쥔 세션은 놓지 못한다)."""
    global _reader
    with _lock:  # 다른 스레드가 만드는 중이면 끝난 뒤에 놓는다
        _reader = None


atexit.register(_release)


def read_lines(image: "Image.Image") -> list[OcrLine]:
    """그림 한 장의 줄들(위→아래, 같은 줄은 왼쪽→오른쪽). 점수로는 거르지 않는다."""
    return get_reader()(image)
