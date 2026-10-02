"""테스트 드라이버 기본 능력 선언."""

from ..vlm import Capabilities

DEFAULT_CAPABILITIES = Capabilities(
    tasks=frozenset({"REGION_TABLE", "REGION_FIGURE", "REGION_TEXT", "PAGE_FULL"}),
    languages=("ko", "en"),
    max_image_pixels=4_000_000,
    max_concurrency=1,
    structured_output=False,
    location="local",
    tier="small",
)
