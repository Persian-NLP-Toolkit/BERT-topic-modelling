"""Persian word cloud generation with optional BERTopic topic weighting."""

from .pipeline import WordCloudResult, compute_frequencies

__all__ = ["WordCloudResult", "compute_frequencies"]
__version__ = "0.2.0"
