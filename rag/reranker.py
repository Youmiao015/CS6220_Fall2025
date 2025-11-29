from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from .config import RerankerConfig
from .llm import TextGenerator


@dataclass
class Candidate:
    payload: dict
    score: float = 0.0


def _format_document(doc: dict) -> str:
    title = doc.get("query_title_en", "")
    content = doc.get("query_content_en", "")
    response = ""
    responses = doc.get("responses") or []
    if responses:
        response = responses[0].get("content_en", "")
    return f"{title}\n{content}\nAnswer: {response}".strip()


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[Candidate], top_k: int) -> List[Candidate]:
        ...


class NoOpReranker(Reranker):
    def rerank(self, query: str, candidates: Sequence[Candidate], top_k: int) -> List[Candidate]:
        return list(candidates)[:top_k]


class MiniLMReranker(Reranker):
    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError("sentence-transformers is required for the MiniLM reranker.") from exc

        self._encoder = CrossEncoder(model_id)

    def rerank(self, query: str, candidates: Sequence[Candidate], top_k: int) -> List[Candidate]:
        texts = [(_format_document(c.payload)) for c in candidates]
        pairs = [[query, doc] for doc in texts]
        scores = self._encoder.predict(pairs)
        reranked = []
        for cand, score in zip(candidates, scores):
            reranked.append(Candidate(payload=cand.payload, score=float(score)))
        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked[:top_k]


class LLMReranker(Reranker):
    TEMPLATE = (
        "You are ranking retrieved documents for a medical visual question answering task.\n"
        "Question:\n{query}\n\n"
        "Documents:\n{documents}\n\n"
        "Return the document IDs in descending relevance order as a comma-separated list."
    )

    def __init__(self, generator: TextGenerator):
        self.generator = generator

    def rerank(self, query: str, candidates: Sequence[Candidate], top_k: int) -> List[Candidate]:
        doc_lines = []
        for idx, cand in enumerate(candidates, start=1):
            text = _format_document(cand.payload)
            doc_lines.append(f"[{idx}] {text}")
        prompt = self.TEMPLATE.format(query=query, documents="\n".join(doc_lines))
        response = self.generator(prompt)
        order = self._parse_order(response, len(candidates))
        ordered = [candidates[i - 1] for i in order if 1 <= i <= len(candidates)]
        if not ordered:
            ordered = list(candidates)
        return ordered[:top_k]

    @staticmethod
    def _parse_order(text: str, total: int) -> List[int]:
        numbers = []
        for token in text.replace("\n", ",").split(","):
            token = token.strip().strip("[]()")
            if not token:
                continue
            if token.isdigit():
                value = int(token)
                if 1 <= value <= total:
                    numbers.append(value)
        return numbers


def build_reranker(config: RerankerConfig, generator: TextGenerator | None) -> Reranker:
    rtype = config.type.lower()
    if rtype == "none":
        return NoOpReranker()
    if rtype in {"crossencoder", "minilm"}:
        model_id = config.model_id or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return MiniLMReranker(model_id=model_id)
    if rtype == "llm":
        if generator is None:
            raise ValueError("LLM reranker requires an LLM generator.")
        return LLMReranker(generator)
    raise ValueError(f"Unknown reranker type: {config.type}")

