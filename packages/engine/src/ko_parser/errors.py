"""엔진 예외. 종료 코드 대응은 cli.py가 정한다."""


class KoParserError(Exception):
    """엔진 예외의 공통 부모."""


class UnsupportedFormat(KoParserError):
    """판별할 수 없거나 파서가 없는 형식."""


class ParseError(KoParserError):
    """파싱 실패. reason은 원인, location은 위치(파일 이름·줄 범위 등)."""

    def __init__(self, reason: str, location: str | None = None) -> None:
        super().__init__(f"{reason} ({location})" if location else reason)
        self.reason = reason
        self.location = location


class DocumentNotFound(KoParserError):
    """저장소에 없는 문서."""


class VersionNotFound(KoParserError):
    """문서는 있지만 그 버전이 없다."""


class StoreConflict(KoParserError):
    """커밋하려는 버전이 저장된 최신 버전 + 1이 아니다."""


class VlmUnavailable(KoParserError):
    """이 엔진에는 VLM 경로가 없다."""


class OcrUnavailable(KoParserError):
    """OCR을 켜라고 했는데 OCR 추가 설치(ko-parser-engine[ocr])가 없거나 깨졌다. 설정 오류(종료 코드 1)."""


class AssetNotFound(KoParserError):
    """저장소에 없는 그림 자산(sha256). asset은 찾은 해시."""

    def __init__(self, asset: str) -> None:
        super().__init__(f"asset not found: {asset}")
        self.asset = asset
