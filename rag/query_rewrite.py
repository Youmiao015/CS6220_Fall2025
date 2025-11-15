from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence

from .config import QueryRewriteConfig
from .llm import TextGenerator


class QueryRewriteStrategy(ABC):
    @abstractmethod
    def generate(self, query: str, context: Optional[dict] = None) -> List[str]:
        ...


class NoOpRewrite(QueryRewriteStrategy):
    def generate(self, query: str, context: Optional[dict] = None) -> List[str]:
        return [query]


class CompositeRewrite(QueryRewriteStrategy):
    def __init__(self, strategies: Sequence[QueryRewriteStrategy]):
        self.strategies = strategies

    def generate(self, query: str, context: Optional[dict] = None) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for strategy in self.strategies:
            variants = strategy.generate(query, context=context)
            for variant in variants:
                text = variant.strip()
                if text and text not in seen:
                    seen.add(text)
                    ordered.append(text)
        if not ordered:
            ordered.append(query)
        return ordered


class HyDERewrite(QueryRewriteStrategy):
    TEMPLATE = (
        "You are a helpful assistant that writes a hypothetical answer passage to help retrieval.\n"
        "Question: {query}\n"
        "Write a concise paragraph that could answer this question based on general medical knowledge."
    )

    def __init__(self, generator: TextGenerator, max_versions: int = 1):
        self.generator = generator
        self.max_versions = max_versions

    def generate(self, query: str, context: Optional[dict] = None) -> List[str]:
        variants = [query]
        for _ in range(self.max_versions):
            prompt = self.TEMPLATE.format(query=query)
            hypo = self.generator(prompt)
            if hypo:
                variants.append(hypo)
        return variants


class DecompositionRewrite(QueryRewriteStrategy):
    TEMPLATE = (
        "Decompose the following medical visual question into smaller sub-questions that can be answered independently.\n"
        "Question: {query}\n"
        "Provide a numbered list of sub-questions."
    )

    def __init__(self, generator: TextGenerator, max_subquestions: int = 3):
        self.generator = generator
        self.max_subquestions = max_subquestions

    def generate(self, query: str, context: Optional[dict] = None) -> List[str]:
        prompt = self.TEMPLATE.format(query=query)
        response = self.generator(prompt)
        variants = [query]
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit():
                line = line.split(".", 1)[-1].strip()
            variants.append(line)
            if len(variants) - 1 >= self.max_subquestions:
                break
        return variants


def build_rewrite_strategy(config: QueryRewriteConfig, generator: TextGenerator | None) -> QueryRewriteStrategy:
    strategy = config.strategy.lower()
    if strategy == "none":
        return NoOpRewrite()
    if strategy == "hyde":
        if generator is None:
            raise ValueError("HyDE rewrite strategy requires an LLM generator.")
        max_versions = int(config.params.get("max_versions", 1))
        return HyDERewrite(generator, max_versions=max_versions)
    if strategy == "decomposition":
        if generator is None:
            raise ValueError("Decomposition strategy requires an LLM generator.")
        max_sub = int(config.params.get("max_subquestions", 3))
        return DecompositionRewrite(generator, max_subquestions=max_sub)
    raise ValueError(f"Unknown query rewrite strategy: {config.strategy}")


def build_rewrite_chain(
    configs: Sequence[QueryRewriteConfig], default_config: QueryRewriteConfig, generator: TextGenerator | None
) -> QueryRewriteStrategy:
    if configs:
        strategies = [build_rewrite_strategy(cfg, generator) for cfg in configs]
        return CompositeRewrite(strategies)
    return build_rewrite_strategy(default_config, generator)

