"""스캔 쪽 OCR 실행부(onnxruntime + numpy + Pillow + pyclipper, 모델은 ko-parser-ocr-models).

OCR 추가 설치(ko-parser-engine[ocr])가 없어도 이 모듈은 import된다: numpy·onnxruntime은 get_reader()가 처음 불릴
때 가져온다(available()은 모듈을 찾기만 한다). 세션(검출·인식)은 프로세스에 하나이고 만들 때만 잠금으로 보호한다.
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
    """OCR 추가 설치에서 찾을 수 없는 모듈 이름. 다 찾히면 None. 찾기만 하고 import하지 않는다: 깔렸는데 import가
    깨지거나 모델 파일이 없는 설치는 '있음'이고 _build가 크게 알린다(자동 모드가 조용히 물러나지 않게)."""
    for module in MODULES:
        if not _installed(module):
            return module
    return None


def available() -> bool:
    """OCR 추가 설치가 있는가(필요한 모듈을 모두 찾을 수 있다). 아무것도 import하지 않고 세션도 만들지 않는다.
    깔렸는데 깨진 설치(import 오류, 모델 파일 없음)도 참이다(get_reader()가 OcrUnavailable)."""
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
    try:  # import는 되는데 쓸 것이 없는 의존성(덜 지운 onnxruntime = 빈 이름공간 패키지 등)도 크게 알린다
        import ko_parser_ocr_models as models

        root = models.model_dir()
        files = (models.DET_FILE, models.REC_FILE, models.DICT_FILE)
        absent = next((root / name for name in files if not (root / name).is_file()), None)
        if absent is None:  # 모델 파일이 없으면 읽개 모듈(onnxruntime)까지 가지 않고 그 파일을 알린다
            from .reader import OcrReader
    except MemoryError:  # 메모리 부족은 설치 문제가 아니다
        raise
    except Exception as exc:
        raise OcrUnavailable(
            f"OCR runtime could not be loaded ({type(exc).__name__}: {exc}); "
            f"reinstall with {INSTALL_HINT} or run with --no-ocr"
        ) from exc
    if absent is not None:  # 모델 패키지는 깔렸는데 파일이 없다: 깨진 설치
        raise OcrUnavailable(f"OCR model file {absent} is missing; reinstall with {INSTALL_HINT} or run with --no-ocr")
    try:
        return OcrReader(root, *files)
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
