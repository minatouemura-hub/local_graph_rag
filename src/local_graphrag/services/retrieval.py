from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from local_graphrag.domain.models import Citation, ContextItem, RetrievalContext, RetrievalHit
from local_graphrag.ports.interfaces import Embedder, FullTextStore, GraphStore, VectorStore


def reciprocal_rank_fusion(*ranked_lists: Sequence[RetrievalHit], k: int = 60) -> list[RetrievalHit]:
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievalHit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            scores[chunk_id] += 1.0 / (k + rank)
            if chunk_id not in best or hit.score > best[chunk_id].score:
                best[chunk_id] = hit
    return [
        RetrievalHit(best[chunk_id].chunk, score, "rrf", rank)
        for rank, (chunk_id, score) in enumerate(
            sorted(scores.items(), key=lambda item: (-item[1], item[0])), start=1
        )
    ]


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector: VectorStore,
        full_text: FullTextStore,
        graph: GraphStore,
        *,
        graph_hops: int = 1,
        graph_max_nodes: int = 20,
    ):
        self.embedder = embedder
        self.vector = vector
        self.full_text = full_text
        self.graph = graph
        self.graph_hops = graph_hops
        self.graph_max_nodes = graph_max_nodes

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
        seed_limit: int | None = None,
    ) -> RetrievalContext:
        if not query.strip() or limit <= 0:
            return RetrievalContext(query, ())
        dense = self.vector.search(self.embedder.embed_query(query), candidate_limit)
        sparse = self.full_text.search(query, candidate_limit)
        fused = reciprocal_rank_fusion(dense, sparse)
        seed_count = seed_limit if seed_limit is not None else max(1, (limit + 1) // 2)
        seeds = fused[: min(limit, seed_count)]
        expanded = self.graph.expand(
            [hit.chunk.chunk_id for hit in seeds],
            hops=self.graph_hops,
            limit=self.graph_max_nodes,
        )
        by_id: dict[str, RetrievalHit] = {hit.chunk.chunk_id: hit for hit in seeds}
        sources: dict[str, set[str]] = {}
        for hit in (*dense, *sparse, *expanded):
            current = by_id.get(hit.chunk.chunk_id)
            if current is None:
                by_id[hit.chunk.chunk_id] = hit
            sources.setdefault(hit.chunk.chunk_id, set()).add(hit.source)
        ordered_ids = [hit.chunk.chunk_id for hit in seeds]
        ordered_ids.extend(hit.chunk.chunk_id for hit in expanded if hit.chunk.chunk_id not in ordered_ids)
        items = tuple(
            ContextItem(
                by_id[chunk_id].chunk,
                by_id[chunk_id].score,
                (Citation(by_id[chunk_id].chunk.document_id, by_id[chunk_id].chunk.page, chunk_id, ""),),
                "+".join(sorted(sources.get(chunk_id, {by_id[chunk_id].source}))),
            )
            for chunk_id in ordered_ids[:limit]
        )
        return RetrievalContext(query, items)
