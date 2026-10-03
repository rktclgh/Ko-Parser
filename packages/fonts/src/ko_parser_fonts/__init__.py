"""ko-parser가 한글 글꼴 없는 리눅스에서 미임베드 한글 글꼴 대신 쓰는 Noto Sans KR(SIL OFL 1.1).

출처: notofonts/noto-cjk 태그 Sans2.004, Sans/SubsetOTF/KR/NotoSansKR-Regular.otf. 라이선스는 fonts/OFL.txt."""

from pathlib import Path

FONT_FILE = "NotoSansKR-Regular.otf"
FONT_SHA256 = "69975a0ac8472717870aefeab0a4d52739308d90856b9955313b2ad5e0148d68"


def font_dir() -> Path:
    """글꼴 파일과 OFL.txt가 든 폴더."""
    return Path(__file__).parent / "fonts"
