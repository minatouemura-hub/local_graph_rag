from __future__ import annotations

from local_graphrag.domain.models import Chunk, ContextItem, GeneratedAnswer, RetrievalContext
from local_graphrag.pipeline.query import QueryPipeline
from local_graphrag.services.context_packer import TokenBudgetContextPacker


class FakeRetriever:
    def __init__(self, context: RetrievalContext) -> None:
        self.context = context

    def retrieve(self, query: str, *, limit: int) -> RetrievalContext:
        return self.context


class FakeGenerator:
    def __init__(self) -> None:
        self.received: RetrievalContext | None = None

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(self, query: str, context: RetrievalContext) -> GeneratedAnswer:
        self.received = context
        return GeneratedAnswer("ok", ())


def test_query_pipeline_packs_context_before_generation() -> None:
    context = RetrievalContext(
        "query",
        tuple(
            ContextItem(Chunk(f"c{i}", "doc", i, "x" * i, None, i), 1.0, ())
            for i in range(1, 4)
        ),
    )
    generator = FakeGenerator()
    pipeline = QueryPipeline(
        FakeRetriever(context),  # type: ignore[arg-type]
        generator,  # type: ignore[arg-type]
        TokenBudgetContextPacker(token_counter=len),
        context_window=135,
        max_output_tokens=1,
        safety_margin_tokens=1,
    )
    answer = pipeline.answer("query", limit=3)
    assert answer.text == "ok"
    assert generator.received is not None
    assert len(generator.received.items) == 2
