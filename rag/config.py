from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str
    text_index: str
    text_metadata: str
    image_index: str
    image_metadata: str
    json_dir: str
    images_root: str

    def resolve(self, base_dir: Path) -> "DatasetConfig":
        base = base_dir.resolve()
        return DatasetConfig(
            name=self.name,
            text_index=str((base / self.text_index).resolve()),
            text_metadata=str((base / self.text_metadata).resolve()),
            image_index=str((base / self.image_index).resolve()),
            image_metadata=str((base / self.image_metadata).resolve()),
            json_dir=str((base / self.json_dir).resolve()),
            images_root=str((base / self.images_root).resolve()),
        )


@dataclass
class LLMConfig:
    provider: str = "transformers"
    model_id: Optional[str] = None
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    hf_token: Optional[str] = None
    task: str = "text-generation"


@dataclass
class QueryRewriteConfig:
    strategy: str = "none"  # none | hyde | decomposition
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankerConfig:
    type: str = "none"  # none | llm | crossencoder
    model_id: Optional[str] = None
    top_k: int = 5
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    dataset: DatasetConfig
    rewrite: QueryRewriteConfig = field(default_factory=QueryRewriteConfig)
    rewrites: List[QueryRewriteConfig] = field(default_factory=list)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: Optional[LLMConfig] = None


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    dataset_cfg = DatasetConfig(**raw["dataset"]).resolve(config_path.parent)
    rewrite_cfg = QueryRewriteConfig(**raw.get("rewrite", {}))
    rewrite_list = [QueryRewriteConfig(**cfg) for cfg in raw.get("rewrites", [])]
    reranker_cfg = RerankerConfig(**raw.get("reranker", {}))
    llm_cfg = raw.get("llm")
    llm = LLMConfig(**llm_cfg) if llm_cfg else None

    return PipelineConfig(
        dataset=dataset_cfg,
        rewrite=rewrite_cfg,
        rewrites=rewrite_list,
        reranker=reranker_cfg,
        llm=llm,
    )

