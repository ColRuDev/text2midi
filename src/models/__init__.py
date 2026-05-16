"""
Vendored models for text2midi generation.

This package contains vendored implementations of external models
that are not available through standard package managers or require
specific configurations.
"""

from .transformer_model import (
    MultiHeadSelfAttention,
    PositionalEncoding,
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
)

__all__ = [
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "PositionalEncoding",
    "MultiHeadSelfAttention",
]
