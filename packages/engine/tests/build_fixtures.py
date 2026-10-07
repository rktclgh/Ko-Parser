"""엔진 골든 예제 생성기(합성 한국어 샘플만).

    uv run python packages/engine/tests/build_fixtures.py          # fixtures/ 재생성
    uv run python packages/engine/tests/build_fixtures.py --check  # 최신인지 확인

입력 샘플의 바이트(UTF-8 BOM·CRLF, cp949)도 여기서 만든다. 저장소의 fixtures/는 .gitattributes로 줄바꿈 변환을 끈다.
"""

import argparse
import codecs
import hashlib
import json
import sys
from pathlib import Path

from hanji.core import build_tree
from hanji.export import to_markdown
from hanji.formats.markdown import MarkdownParser
from hanji_contracts import DocumentTree, SourceInfo

ROOT = Path(__file__).resolve().parent / "fixtures"

# 빈 목록 항목 "- "의 뒤 공백은 의미가 있어 \x20으로 명시한다(편집기·git diff --check가 지우지 않도록).
REPORT = """# 2026년 사업 계획

본 문서는 **2026년** 사업 [추진 현황](https://example.com)을 정리한다.
둘째 줄은 `예산 코드`를 포함한다.\\
강제 줄바꿈 뒤 글.

## 1. 추진 일정

- 1분기: 참여 기관 모집
  - 공고문 게시
- 2분기: 선정 결과 통보
-\x20

3. 셋째 단계
4. 넷째 단계

![추진 체계도](org.png)

> 참고: 세부 일정은 *변경될 수 있다*.

## 2. 예산

| 분류 | 금액(원) |
|---|---|
| *인건비* | 4,250,000 |
| 운영비 | 1\\|800,000 |
| 예비비 |

```text
총액 = 인건비 + 운영비
```

<div>
별첨 1. 산출 근거
</div>

---

붙임
====

자세한 내용은 별첨을 참고한다.

자세한 내용은 별첨을 참고한다.
"""

MEMO = """# 회의록

일시: 2026년 10월 2일

- 참석: 기획팀, 예산팀
- 안건: 하반기 집행 점검

## 결정 사항

1. 집행률 보고는 매월 한다.
"""

SAMPLES: dict[str, bytes] = {
    "report.md": REPORT.encode("utf-8"),
    "memo_bom_crlf.md": codecs.BOM_UTF8 + MEMO.replace("\n", "\r\n").encode("utf-8"),
    "memo_cp949.md": MEMO.encode("cp949"),
    "empty.md": b"",
}


def document_id(name: str) -> str:
    return "fx-md-" + Path(name).stem


def golden_tree(name: str, data: bytes) -> DocumentTree:
    source = SourceInfo(name=name, mime="text/markdown", content_hash="sha256:" + hashlib.sha256(data).hexdigest())
    return build_tree(MarkdownParser().parse(data, name), document_id(name), 1, source)


def build_all() -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for name, data in SAMPLES.items():
        stem = Path(name).stem
        tree = golden_tree(name, data)
        out[f"inputs/{name}"] = data
        out[f"expected/{stem}.json"] = (
            json.dumps(tree.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        out[f"expected/{stem}.md"] = to_markdown(tree).encode("utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="hanji fixtures")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    stale = []
    for rel, data in build_all().items():
        path = ROOT / rel
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                stale.append(rel)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print(path)
    for rel in stale:
        print(f"stale fixture: {rel}", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
