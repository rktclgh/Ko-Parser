"""순수 core: 저장·입출력 없이 트리를 만들고 비교한다."""

from .build import build_tree
from .diff import diff_trees

__all__ = ["build_tree", "diff_trees"]
