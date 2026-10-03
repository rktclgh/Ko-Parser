"""파싱 결과 뷰어: 인터넷 없이 열리는 HTML 한 장."""

from .html import render_html
from .images import DEFAULT_DPI, render_page_images

__all__ = ["render_html", "render_page_images", "DEFAULT_DPI"]
