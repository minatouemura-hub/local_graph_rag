from __future__ import annotations

import math
import re
from collections.abc import Callable

from local_graphrag.domain.models import Chunk, ContextItem, RetrievalContext

TokenCounter = Callable[[str], int]


def estimate_token_count(text: str) -> int:
    """Conservative tokenizer-free estimate for mixed Japanese/English text."""
    if not text:
        return 0
    non_ascii = sum(ord(char) > 127 for char in text)
    ascii_chars = len(text) - non_ascii
    estimate = math.ceil(non_ascii / 2) + math.ceil(ascii_chars / 4)
    estimate += len(re.findall(r"\s+", text))
    return max(1, estimate)


class TokenBudgetContextPacker:
    def __init__(self, token_counter: TokenCounter = estimate_token_count) -> None:
        self.token_counter = token_counter

    def pack(self, query: str, context: RetrievalContext, token_budget: int) -> RetrievalContext:
        if token_budget <= 0 or not context.items:
            return RetrievalContext(query, ())
        selected: list[ContextItem] = []
        used = 0
        seen: set[str] = set()
        for item in context.items:
            if item.chunk.chunk_id in seen:
                continue
            item_tokens = self.token_counter(item.chunk.text)
            if used + item_tokens <= token_budget:
                selected.append(item)
                seen.add(item.chunk.chunk_id)
                used += item_tokens
                continue
            if not selected and item_tokens > token_budget:
                truncated = self._truncate(item, token_budget)
                if truncated is not None:
                    selected.append(truncated)
            break
        return RetrievalContext(query, tuple(selected))

    def _truncate(self, item: ContextItem, token_budget: int) -> ContextItem | None:
        if not item.chunk.text or token_budget <= 0:
            return None
        low, high = 1, len(item.chunk.text)
        best = ""
        while low <= high:
            middle = (low + high) // 2
            candidate = item.chunk.text[:middle].rstrip()
            if self.token_counter(candidate) <= token_budget:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        if not best:
            return None
        metadata = dict(item.chunk.metadata)
        metadata["packed_truncated"] = True
        chunk = Chunk(
            chunk_id=item.chunk.chunk_id,
            document_id=item.chunk.document_id,
            page=item.chunk.page,
            text=best,
            section_id=item.chunk.section_id,
            order=item.chunk.order,
            bbox=item.chunk.bbox,
            layout_types=item.chunk.layout_types,
            metadata=metadata,
        )
        return ContextItem(chunk, item.score, item.citations, f"{item.reason}+truncated")
