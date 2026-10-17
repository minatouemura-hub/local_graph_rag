from __future__ import annotations

from local_graphrag.domain.models import Chunk, ContextItem, RetrievalContext
from local_graphrag.services.context_packer import TokenBudgetContextPacker


def make_context() -> RetrievalContext:
    items = tuple(
        ContextItem(
            Chunk(f"c{i}", "doc", i, "x" * i, None, i),
            1.0 / i,
            (),
            f"dense:{i}",
        )
        for i in range(1, 4)
    )
    return RetrievalContext("query", items)


def test_packer_keeps_rank_order_and_fits_budget() -> None:
    context = make_context()
    packed = TokenBudgetContextPacker(token_counter=len).pack("query", context, 3)
    assert [item.chunk.chunk_id for item in packed.items] == ["c1", "c2"]
    assert sum(len(item.chunk.text) for item in packed.items) <= 3


def test_packer_preserves_citations_and_reason() -> None:
    context = make_context()
    packed = TokenBudgetContextPacker(token_counter=len).pack("query", context, 1)
    assert packed.items[0].citations == ()
    assert packed.items[0].reason == "dense:1"


def test_packer_handles_zero_budget_and_truncates_oversized_first_chunk() -> None:
    context = RetrievalContext(
        "query",
        (ContextItem(Chunk("large", "doc", 1, "xxxxx", None, 0), 1.0, (), "dense"),),
    )
    packer = TokenBudgetContextPacker(token_counter=len)
    assert packer.pack("query", context, 0).items == ()
    packed = packer.pack("query", context, 2)
    assert packed.items[0].chunk.text == "xx"
    assert packed.items[0].chunk.metadata["packed_truncated"] is True
    assert "truncated" in packed.items[0].reason
