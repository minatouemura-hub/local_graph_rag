from __future__ import annotations

from local_graphrag.domain.models import GeneratedAnswer
from local_graphrag.ports.interfaces import Generator
from local_graphrag.services.context_packer import TokenBudgetContextPacker, estimate_token_count
from local_graphrag.services.retrieval import HybridRetriever


class QueryPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator,
        context_packer: TokenBudgetContextPacker | None = None,
        context_window: int = 4096,
        max_output_tokens: int = 256,
        safety_margin_tokens: int = 1024,
    ):
        self.retriever, self.generator = retriever, generator
        self.context_packer = context_packer
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.safety_margin_tokens = safety_margin_tokens

    def answer(self, query: str, *, limit: int = 5) -> GeneratedAnswer:
        context = self.retriever.retrieve(query, limit=limit)
        if self.context_packer is not None:
            budget = (
                self.context_window
                - self.max_output_tokens
                - self.safety_margin_tokens
                - estimate_token_count(query)
                - 128
            )
            context = self.context_packer.pack(query, context, budget)
        self.generator.load()
        try:
            return self.generator.generate(query, context)
        finally:
            self.generator.unload()
