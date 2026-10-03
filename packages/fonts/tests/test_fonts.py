"""번들 글꼴이 고정한 파일 그대로이고 라이선스가 함께 있는지."""

import hashlib

import ko_parser_fonts


def test_font_file_is_the_pinned_noto_sans_kr():
    path = ko_parser_fonts.font_dir() / ko_parser_fonts.FONT_FILE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ko_parser_fonts.FONT_SHA256


def test_ofl_license_ships_with_the_font():
    text = (ko_parser_fonts.font_dir() / "OFL.txt").read_text(encoding="utf-8")
    assert text.startswith("This Font Software is licensed under the SIL Open Font License")


def test_font_dir_holds_only_the_font_and_license():
    """PDFium은 이 폴더를 통째로 훑는다: 다른 파일이 섞이면 글꼴 고르기가 달라질 수 있다."""
    names = sorted(p.name for p in ko_parser_fonts.font_dir().iterdir() if not p.name.startswith("."))  # .DS_Store 등
    assert names == [ko_parser_fonts.FONT_FILE, "OFL.txt"]
