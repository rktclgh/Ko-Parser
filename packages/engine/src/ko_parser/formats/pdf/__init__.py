"""PDF 파서(텍스트 레이어). 레이아웃 모델 없이 글자 위치·크기로 블록을 묶는다."""

from .parser import PdfParser

__all__ = ["PdfParser"]
