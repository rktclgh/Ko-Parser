"""한글 글꼴 없는 리눅스에서 hanji-fonts의 Noto Sans KR을 PDFium 글꼴 검색 경로에 더한다.

PDFium은 미임베드 한글 글꼴을 시스템 글꼴로 대신 그리는데, 한글 글꼴이 없으면 한 글자짜리 글자 객체가 텍스트
쪽에서 빠진다(ubuntu-ci-debug-report). 리눅스 PDFium은 초기화 때 받은 사용자 글꼴 경로만 훑으므로(기본 경로를
대신한다) 기본 경로에 번들 폴더를 더해 다시 초기화한다. macOS·Windows는 OS 글꼴을 쓰고 사용자 경로를 보지 않는다.
패키지가 없거나 다시 초기화할 수 없으면(이미 열린 PDFium 객체가 있다) 아무것도 하지 않고 extract의 가드가 지킨다.
PDFIUM_LOCK 없이 다른 스레드에서 pypdfium2를 부르는 코드는 다시 초기화와 겹칠 수 있다. hanji는 늘 잠금을 잡는다."""

import ctypes
import gc
import os
import sys
from pathlib import Path

import pypdfium2.internal as pdfium_i
import pypdfium2.raw as pdfium_c

# PDFium CFX_LinuxFontInfo의 기본 검색 경로(사용자 경로를 주면 이 목록이 빠지므로 함께 넘긴다)
LINUX_FONT_PATHS = ("/usr/share/fonts", "/usr/share/X11/fonts/Type1", "/usr/share/X11/fonts/TTF", "/usr/local/share/fonts")

_attempted = False  # 프로세스당 한 번만 시도한다
_registered = False
_keep_alive: list[object] = []  # PDFium에 넘긴 C 문자열 배열


def font_paths(extra: Path) -> tuple[bytes, ...]:
    return (*(p.encode() for p in LINUX_FONT_PATHS), os.fsencode(extra))


def _live_pdfium_objects() -> bool:
    return any(pdfium_i.ObjectTracker.values())


def register_bundled_fonts() -> bool:
    """PDFIUM_LOCK을 잡은 채로 부른다. 이 프로세스에서 번들 글꼴을 등록했으면 True."""
    global _attempted, _registered
    if _attempted:
        return _registered
    if not sys.platform.startswith("linux"):
        _attempted = True
        return False
    try:
        import hanji_fonts
    except ImportError:  # 패키지가 없다: 가드가 설치를 안내한다. 설치됐는데 깨졌으면 아래에서 그대로 실패해 드러낸다
        _attempted = True
        return False
    bundle = hanji_fonts.font_dir()
    if _live_pdfium_objects():
        gc.collect()  # 참조 순환에만 남아 아직 닫히지 않은 문서는 모으면 닫힌다
        if _live_pdfium_objects():
            return False  # 시도로 치지 않는다: 열린 객체가 없어지면 다음 open_pdf에서 등록한다
    paths = (ctypes.c_char_p * (len(LINUX_FONT_PATHS) + 2))(*font_paths(bundle), None)
    _keep_alive.append(paths)
    config = pdfium_c.FPDF_LIBRARY_CONFIG(
        version=2,
        m_pUserFontPaths=ctypes.cast(paths, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
        m_pIsolate=None,
        m_v8EmbedderSlot=0,
    )
    pdfium_c.FPDF_DestroyLibrary()
    pdfium_c.FPDF_InitLibraryWithConfig(config)
    _attempted = _registered = True
    return True
