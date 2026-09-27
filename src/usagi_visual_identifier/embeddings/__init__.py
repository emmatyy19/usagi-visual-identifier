"""Frozen image embeddings and exact retrieval tools."""

from usagi_visual_identifier.embeddings.aggregation import (
    DEFAULT_AGGREGATION_CONFIG,
    AggregationConfig,
    AggregationMethod,
    IdentityMatch,
    rank_gallery_identities,
)
from usagi_visual_identifier.embeddings.artifacts import (
    EmbeddingArtifact,
    EmbeddingBuildResult,
    EmbeddingExample,
    build_embedding_artifact,
    load_embedding_artifact,
    load_prepared_examples,
    select_embedding_examples,
)
from usagi_visual_identifier.embeddings.dinov2 import (
    DEFAULT_DINOV2_MODEL_ID,
    DEFAULT_DINOV2_REVISION,
    Dinov2Encoder,
    select_torch_device,
)
from usagi_visual_identifier.embeddings.evaluation import (
    AggregationComparisonReport,
    RetrievalReport,
    compare_retrieval_aggregations,
    evaluate_retrieval,
    write_aggregation_comparison,
    write_retrieval_report,
)
from usagi_visual_identifier.embeddings.inference import (
    FrozenRetrievalPredictor,
    ImagePrediction,
    LatencySummary,
    load_query_image,
)

__all__ = [
    "DEFAULT_AGGREGATION_CONFIG",
    "DEFAULT_DINOV2_MODEL_ID",
    "DEFAULT_DINOV2_REVISION",
    "AggregationComparisonReport",
    "AggregationConfig",
    "AggregationMethod",
    "Dinov2Encoder",
    "EmbeddingArtifact",
    "EmbeddingBuildResult",
    "EmbeddingExample",
    "FrozenRetrievalPredictor",
    "IdentityMatch",
    "ImagePrediction",
    "LatencySummary",
    "RetrievalReport",
    "build_embedding_artifact",
    "compare_retrieval_aggregations",
    "evaluate_retrieval",
    "load_embedding_artifact",
    "load_prepared_examples",
    "load_query_image",
    "rank_gallery_identities",
    "select_embedding_examples",
    "select_torch_device",
    "write_aggregation_comparison",
    "write_retrieval_report",
]
