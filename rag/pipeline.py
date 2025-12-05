from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from PIL import Image

from retriever import Retriever, RetrieverConfig

from .config import PipelineConfig
from .llm import build_generator
from .query_rewrite import QueryRewriteStrategy, build_rewrite_chain
from .reranker import Candidate, Reranker, build_reranker


@dataclass
class RetrievalPipeline:
    retriever: Retriever
    rewrite_strategy: QueryRewriteStrategy
    reranker: Reranker
    max_candidates: int = 8

    def retrieve(self, query: str, query_image: Optional[Image.Image] = None, top_k: int = 2) -> List[dict]:
        rewritten_queries = self.rewrite_strategy.generate(query)
        candidates: List[Candidate] = []
        seen_ids: Set[str] = set()

        for rewritten in rewritten_queries:
            # Use hybrid retrieval if image provided, otherwise text-only
            if query_image is not None:
                results = self.retriever.retrieve_hybrid(
                    rewritten, query_image, top_k=self.max_candidates, alpha=0.5
                )
            else:
                results = self.retriever.retrieve_by_text(rewritten, top_k=self.max_candidates)

            for item in results:
                eid = item.get("encounter_id")
                if not eid or eid in seen_ids:
                    continue
                seen_ids.add(eid)
                candidates.append(Candidate(payload=item))

        if not candidates:
            return []

        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        return [cand.payload for cand in reranked]


def build_pipeline(config: PipelineConfig) -> RetrievalPipeline:
    retriever = Retriever(
        RetrieverConfig(
            text_index=config.dataset.text_index,
            text_metadata=config.dataset.text_metadata,
            image_index=config.dataset.image_index,
            image_metadata=config.dataset.image_metadata,
        )
    )

    generator = build_generator(config.llm)
    rewrite_strategy = build_rewrite_chain(config.rewrites, config.rewrite, generator)
    reranker = build_reranker(config.reranker, generator)

    return RetrievalPipeline(
        retriever=retriever,
        rewrite_strategy=rewrite_strategy,
        reranker=reranker,
        max_candidates=max(config.reranker.top_k * 3, 6),
    )

