"""Frozen image embeddings and exact retrieval tools."""

from usagi_visual_identifier.embeddings.artifacts import (
    EmbeddingArtifact,
    EmbeddingBuildResult,
    EmbeddingExample,
    build_embedding_artifact,
    load_embedding_artifact,
    load_prepared_examples,
)
from usagi_visual_identifier.embeddings.dinov2 import (
    DEFAULT_DINOV2_MODEL_ID,
    DEFAULT_DINOV2_REVISION,
    Dinov2Encoder,
    select_torch_device,
)
from usagi_visual_identifier.embeddings.evaluation import (
    RetrievalReport,
    evaluate_retrieval,
    write_retrieval_report,
)

__all__ = [
    "DEFAULT_DINOV2_MODEL_ID",
    "DEFAULT_DINOV2_REVISION",
    "Dinov2Encoder",
    "EmbeddingArtifact",
    "EmbeddingBuildResult",
    "EmbeddingExample",
    "RetrievalReport",
    "build_embedding_artifact",
    "evaluate_retrieval",
    "load_embedding_artifact",
    "load_prepared_examples",
    "select_torch_device",
    "write_retrieval_report",
]
