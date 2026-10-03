"""리눅스에서 번들 한글 글꼴을 PDFium 글꼴 경로에 더하는지. 등록 판단은 초기화 함수를 기록기로 바꿔 시험하고,
실제 글자 결과는 새 PDFium을 쓰는 하위 프로세스에서 시험한다(리눅스의 테스트 프로세스도 다른 PDF 테스트가
open_pdf를 부르면서 한 번 다시 초기화된다)."""

import io
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from ko_parser.errors import ParseError
from ko_parser.formats.pdf import fonts


@pytest.fixture
def stubbed(monkeypatch):
    """등록 상태를 비우고, PDFium 초기화 함수를 기록기로 바꾼다. 열린 객체 검사는 실제 것을 쓴다."""
    calls = []
    monkeypatch.setattr(fonts, "_attempted", False)
    monkeypatch.setattr(fonts, "_registered", False)
    monkeypatch.setattr(fonts.pdfium_c, "FPDF_DestroyLibrary", lambda: calls.append("destroy"))
    monkeypatch.setattr(fonts.pdfium_c, "FPDF_InitLibraryWithConfig", lambda cfg: calls.append("init"))
    monkeypatch.setattr(fonts.sys, "platform", "linux")
    return calls


@pytest.fixture
def fresh(stubbed, monkeypatch):
    """stubbed에 더해 열린 PDFium 객체가 없다고 본다."""
    monkeypatch.setattr(fonts, "_live_pdfium_objects", lambda: False)
    return stubbed


def test_font_paths_keep_pdfium_linux_defaults_and_add_the_bundle():
    extra = Path("/opt/글꼴 폴더")
    paths = fonts.font_paths(extra)
    assert paths[:-1] == tuple(p.encode() for p in fonts.LINUX_FONT_PATHS)
    assert paths[-1] == os.fsencode(extra)  # 같은 Path로 기대값을 만든다(Windows는 구분자가 \\로 바뀐다)


def test_register_reinitializes_pdfium_once_on_linux(fresh):
    assert fonts.register_bundled_fonts() is True
    assert fonts.register_bundled_fonts() is True
    assert fresh == ["destroy", "init"]


def test_register_does_nothing_off_linux(fresh, monkeypatch):
    monkeypatch.setattr(fonts.sys, "platform", "darwin")
    assert fonts.register_bundled_fonts() is False
    assert fresh == []


def test_register_skips_when_pdfium_objects_are_open(fresh, monkeypatch):
    """다른 코드가 연 pypdfium2 문서가 있으면 다시 초기화하면 그 문서가 깨진다: 건너뛰고 가드에 맡긴다.
    건너뛴 것은 '시도'로 치지 않는다: 열린 객체가 없어진 다음 호출에서 등록한다."""
    monkeypatch.setattr(fonts, "_live_pdfium_objects", lambda: True)
    assert fonts.register_bundled_fonts() is False
    assert fresh == []
    monkeypatch.setattr(fonts, "_live_pdfium_objects", lambda: False)
    assert fonts.register_bundled_fonts() is True
    assert fresh == ["destroy", "init"]


def one_page_pdf() -> bytes:
    from reportlab.pdfgen.canvas import Canvas
    b = io.BytesIO()
    c = Canvas(b, invariant=1)
    c.showPage()
    c.save()
    return b.getvalue()


def test_live_pdfium_objects_tracks_real_documents():
    """pypdfium2가 실제로 연 문서를 ObjectTracker로 보는지(모든 OS)."""
    pdf = pdfium.PdfDocument(one_page_pdf())
    assert fonts._live_pdfium_objects() is True
    pdf.close()
    assert fonts._live_pdfium_objects() is False


def test_register_collects_unreachable_documents_first(stubbed):
    """참조 순환에만 남은 문서는 gc가 아직 닫지 않았을 뿐이다: 한 번 모아 보고 등록한다."""
    pdf = pdfium.PdfDocument(one_page_pdf())
    pdf.self_ref = pdf  # 순환: del로는 풀리지 않고 gc만 닫는다
    del pdf
    assert fonts._live_pdfium_objects() is True
    assert fonts.register_bundled_fonts() is True
    assert stubbed == ["destroy", "init"]


def test_register_surfaces_a_broken_fonts_package(fresh, monkeypatch):
    """설치는 됐는데 깨진 패키지를 조용히 건너뛰면 가드가 '설치하라'고 잘못 안내한다: 그대로 실패해 드러낸다."""
    monkeypatch.setitem(sys.modules, "ko_parser_fonts", types.SimpleNamespace())  # font_dir 없음
    with pytest.raises(AttributeError):
        fonts.register_bundled_fonts()
    assert fresh == []


def test_open_pdf_registers_bundled_fonts_first(monkeypatch):
    """모든 OS에서 open_pdf가 등록을 부르는지(리눅스 밖에서는 등록이 아무 일도 안 해 연결이 빠져도 모른다)."""
    from ko_parser.formats.pdf import extract
    calls = []
    monkeypatch.setattr(fonts, "register_bundled_fonts", lambda: calls.append("register") or False)
    with extract.PDFIUM_LOCK, pytest.raises(ParseError):
        extract.open_pdf(b"not a pdf", "x.pdf")
    assert calls == ["register"]


def test_register_skips_without_the_fonts_package(fresh, monkeypatch):
    monkeypatch.setitem(sys.modules, "ko_parser_fonts", None)  # import가 ImportError
    assert fonts.register_bundled_fonts() is False
    assert fresh == []


SINGLE_GLYPH = textwrap.dedent("""
    import io, sys
    {block}
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen.canvas import Canvas
    from ko_parser.errors import ParseError
    from ko_parser.formats.pdf.extract import extract_pages
    pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
    b = io.BytesIO(); c = Canvas(b, invariant=1); t = c.beginText(72, 770); t.setFont("HYGothic-Medium", 18)
    t.textOut("가"); c.drawText(t); c.showPage(); c.save()
    try:  # ASCII만 출력한다: Windows 파이프의 stdout은 ANSI 코드 페이지라 한글을 못 쓴다
        print(" ".join(f"U+{{ord(ch.text):04X}}" for ch in extract_pages(b.getvalue(), "t.pdf")[0].chars))
    except ParseError as exc:
        print("ParseError:", ascii(str(exc)))
""")


def run_single_glyph(block: str = "") -> str:
    code = SINGLE_GLYPH.format(block=block)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_single_glyph_hangul_object_is_read():
    """한글 글꼴 없는 리눅스(CI ubuntu)에서는 번들 글꼴 덕에, macOS·Windows는 시스템 글꼴로 '가'가 읽힌다."""
    assert run_single_glyph() == "U+AC00"  # '가'


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="리눅스에서만 번들 글꼴을 쓴다")
def test_without_bundled_fonts_linux_still_raises():
    """번들 패키지가 없으면 지금처럼 가드가 ParseError로 알린다(시스템에 한글 글꼴이 있으면 시험할 수 없다).
    CI ubuntu는 KO_PARSER_CI_NO_KOREAN_FONT=1을 주므로 거기서는 건너뛰지 않고 실패한다(검증이 조용히 사라지지 않게)."""
    out = run_single_glyph('sys.modules["ko_parser_fonts"] = None')
    if out == "U+AC00":
        if os.environ.get("KO_PARSER_CI_NO_KOREAN_FONT") == "1":
            pytest.fail("CI ubuntu에 시스템 한글 글꼴이 생겼다: 번들 없음 가드 경로를 시험할 수 없다")
        pytest.skip("이 리눅스에는 시스템 한글 글꼴이 있다")
    assert out.startswith("ParseError:") and "ko-parser-engine[fonts]" in out
