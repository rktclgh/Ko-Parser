"""두 버전의 블록 차이 → 계약 DocumentChange. lineage는 만들지 않는다(D5)."""

from ko_parser_contracts import DocumentChange, DocumentTree


def diff_trees(prev: DocumentTree | None, new: DocumentTree) -> DocumentChange | None:
    """prev가 None이면 첫 버전(블록이 없어도 변경 1건). 그 밖에 added·updated·removed가 모두 비고 쪽 정보(pages)도
    같으면 None. 블록은 같고 쪽 정보만 바뀌면(쪽 판정 등) 빈 변경으로 새 버전을 만든다.

    updated는 공통 id 중 블록 필드가 하나라도 바뀐 것: order·locator·section_path·state·confidence·text_source·
    region_id, 그리고 정규화 해시가 같아 id가 유지된 원문 text·table 차이. 그림은 이미지 해시(figure.asset)만 블록 해시에 들어가
    이미지가 바뀌거나 생기거나 없어지면 블록이 바뀌고(removed + added), 짝 캡션(caption_block_id)·dpi·분류(category)
    등 나머지 figure 필드만 바뀌면 updated다.
    순서: added·updated는 새 트리 순서, removed는 이전 트리 순서.
    """
    if prev is None:
        return DocumentChange(document_id=new.document_id, version=new.version,
                              added=tuple(b.block_id for b in new.blocks))
    if prev.document_id != new.document_id:
        raise ValueError("document_id mismatch")
    old = {b.block_id: b for b in prev.blocks}
    new_ids = {b.block_id for b in new.blocks}
    added = tuple(b.block_id for b in new.blocks if b.block_id not in old)
    updated = tuple(b.block_id for b in new.blocks if b.block_id in old and old[b.block_id] != b)
    removed = tuple(b.block_id for b in prev.blocks if b.block_id not in new_ids)
    if not (added or updated or removed) and prev.pages == new.pages:
        return None
    return DocumentChange(document_id=new.document_id, version=new.version, previous_version=prev.version,
                          added=added, updated=updated, removed=removed)
