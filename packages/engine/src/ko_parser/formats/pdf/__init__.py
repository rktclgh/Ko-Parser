"""PDF 파서(텍스트 레이어 + 선택 OCR·레이아웃 모델). 글자 위치·크기로 블록을 묶고 그림·캡션은 이미지 객체와 레이아웃 모델로 찾는다."""

from .parser import PdfParser

__all__ = ["PdfParser"]
