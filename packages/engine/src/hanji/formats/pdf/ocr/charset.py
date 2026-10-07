"""인식 글자 목록(스펙 §5). PaddlePaddle 공식 한국어 인식 모델은 따로 된 사전 없이 설정 파일(inference.yml)의
PostProcess.character_dict에 글자 목록을 둔다. PyYAML 없이 그 블록 목록만 읽는다: 이 파일에 나오는 두 모양, 그대로 쓴
글자(`  - 가`, `  - \\` 포함)와 작은따옴표로 감싼 글자(`  - '#'`, 안의 `''`는 `'`)만 다룬다. 읽은 목록을 줄마다 한 글자 +
끝 줄바꿈으로 이은 바이트(예전 dict.txt와 같다)의 SHA-256을 고정해 읽기가 틀리면 설정 오류로 알린다. 표준 라이브러리만."""

import hashlib
from pathlib import Path

from ....errors import ModelError

DICT_SHA256 = "a88071c68c01707489baa79ebe0405b7beb5cca229f4fc94cc3ef992328802d7"  # "\n".join(글자) + "\n", 11,945자
_KEY = "  character_dict:"
_ITEM = "  - "


def read_character_dict(text: str) -> list[str]:
    """설정 글에서 `  character_dict:` 다음에 이어지는 `  - ` 줄들의 값. 작은따옴표로 감싼 값은 벗기고 `''`를 `'`로.
    블록이 없으면 빈 목록. 줄은 LF로만 나눈다(글자 목록에 U+2028 같은 줄 구분 문자가 있어도 깨지지 않게)."""
    out: list[str] = []
    inside = False
    for line in text.split("\n"):
        if line == _KEY:
            inside = True
        elif inside and line.startswith(_ITEM):
            value = line[len(_ITEM):]
            if len(value) >= 2 and value[0] == value[-1] == "'":
                value = value[1:-1].replace("''", "'")
            out.append(value)
        elif inside:
            break
    return out


def load_symbols(config: Path) -> list[str]:
    """인식 모델 설정 파일(models.resolve()가 크기·SHA-256을 확인한 것)의 글자 목록. 이은 바이트의 SHA-256이
    DICT_SHA256과 다르면 ModelError(설정 오류). 확인을 지난 파일이라 다시 받아도 같다: 이 빌드의 models.toml과
    DICT_SHA256이 서로 맞지 않는 것이다."""
    symbols = read_character_dict(config.read_text(encoding="utf-8"))
    if hashlib.sha256(("\n".join(symbols) + "\n").encode("utf-8")).hexdigest() != DICT_SHA256:
        raise ModelError(f"character list in {config} ({len(symbols)} symbols) does not match the pinned SHA-256: "
                         f"this hanji build pins a recognition config (models.toml) and a character list "
                         f"(charset.DICT_SHA256) that disagree; reinstall hanji or report it")
    return symbols
