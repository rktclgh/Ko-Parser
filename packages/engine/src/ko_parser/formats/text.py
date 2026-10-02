"""텍스트 디코딩: UTF-8(BOM 제거) → cp949 → ParseError. 줄바꿈 \r\n·\r은 \n으로."""

from ..errors import ParseError

ENCODINGS = ("utf-8-sig", "cp949")


def decode_text(data: bytes, name: str) -> str:
    for encoding in ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        return text.replace("\r\n", "\n").replace("\r", "\n")
    raise ParseError("cannot decode text as UTF-8 or cp949", name)
