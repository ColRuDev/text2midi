"""
Vendored Transformer Model for Text2MIDI.

This module contains a custom Encoder-Decoder Transformer architecture
specifically designed for text-to-MIDI generation. The encoder uses a
pretrained T5 model, while the decoder is a custom transformer decoder.

Source: https://github.com/amaai-lab/text2midi
License: MIT

This vendored version includes only the essential components for inference.
"""

from __future__ import annotations

import copy
import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from transformers import T5EncoderModel

__all__ = [
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "PositionalEncoding",
    "MultiHeadSelfAttention",
]


def _generate_square_subsequent_mask(
    sz: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions
    are filled with float(0.0).
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        return None
    src_size = src.size()
    if len(src_size) == 2:
        return src_size[0]
    else:
        seq_len_pos = 1 if batch_first else 0
        return src_size[seq_len_pos]


def _get_clones(module: Module, n: int) -> ModuleList:
    """Return N copies of a module."""
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal."""
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype
        )
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


class PositionalEncoding(nn.Module):
    """Inject positional information into the input embeddings.

    Uses sine and cosine functions of different frequencies.

    Args:
        d_model: The embedding dimension.
        dropout: The dropout value (default=0.1).
        max_len: The max length of the incoming sequence (default=5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape [sequence length, batch size, embed dim].

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module with optional causal masking.

    Args:
        embed_dim: The input embedding dimension.
        num_heads: The number of attention heads (default: 4).
        dropout: The dropout probability (default: 0.1).
        batch_first: If True, input is (batch, seq, feature) (default: True).
        device: The device to use.
        dtype: The data type to use.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        batch_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.batch_first = batch_first
        self.dim_head = embed_dim // num_heads
        self.scale = self.dim_head**-0.5
        self.heads = num_heads
        hidden_dim = self.dim_head * num_heads
        self.to_qkv = nn.Linear(embed_dim, hidden_dim * 3, bias=False, **factory_kwargs)
        self.to_out = nn.Linear(hidden_dim, embed_dim, bias=False, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        """Forward pass of multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embed_dim).
            is_causal: If True, apply causal mask (default: True).

        Returns:
            Output tensor of shape (batch_size, sequence_length, embed_dim).
        """
        if not self.batch_first:
            x = x.transpose(0, 1)
        b, n, _ = x.size()
        q, k, v = torch.chunk(self.to_qkv(x), chunks=3, dim=-1)
        q, k, v = map(lambda t: t.contiguous().view(b, self.heads, n, -1), (q, k, v))

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.contiguous().view(b, n, -1)
        out = self.dropout(out)
        return self.to_out(out)


class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: An instance of TransformerDecoderLayer.
        num_layers: The number of sub-decoder-layers in the decoder.
        norm: The layer normalization component (optional).
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """Pass the inputs through the decoder layers.

        Args:
            tgt: The sequence to the decoder.
            memory: The sequence from the last layer of the encoder.
            tgt_mask: The mask for the tgt sequence.
            memory_mask: The mask for the memory sequence.
            memory_key_padding_mask: The mask for memory keys per batch.
            tgt_is_causal: If True, apply causal mask.
            memory_is_causal: If True, apply causal mask for memory.

        Returns:
            Output tensor from the decoder.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                memory,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(Module):
    """TransformerDecoderLayer with self-attention, cross-attention, and feedforward.

    Args:
        d_model: The number of expected features in the input.
        nhead: The number of heads in the multiheadattention models.
        dim_feedforward: The dimension of the feedforward network (default=2048).
        dropout: The dropout value (default=0.1).
        activation: The activation function (default: relu).
        layer_norm_eps: The eps value in layer normalization (default=1e-5).
        batch_first: If True, input is (batch, seq, feature) (default: False).
        norm_first: If True, layer norm is done before attention (default: False).
        bias: If True, Linear and LayerNorm learn bias (default: True).
        device: The device to use.
        dtype: The data type to use.
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )

        # Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """Pass the inputs through the decoder layer.

        Args:
            tgt: The sequence to the decoder layer.
            memory: The sequence from the encoder.
            memory_mask: The mask for the memory sequence.
            memory_key_padding_mask: The mask for memory keys per batch.
            tgt_is_causal: If True, apply causal mask for tgt.
            memory_is_causal: If True, apply causal mask for memory.

        Returns:
            Output tensor.
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_is_causal)
            x = x + self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_is_causal))
            x = self.norm2(
                x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    def _sa_block(self, x: Tensor, is_causal: bool = False) -> Tensor:
        """Self-attention block."""
        x = self.self_attn(x, is_causal=is_causal)
        return self.dropout1(x)

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        """Multi-head attention block."""
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feedforward block."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class Transformer(Module):
    """Custom Transformer model for text-to-MIDI generation.

    This model uses a pretrained T5 encoder for text understanding and a
    custom transformer decoder for MIDI token generation.

    Args:
        n_vocab: The vocabulary size for MIDI tokens (default=30000).
        d_model: The number of expected features (default=512).
        nhead: The number of heads in the multiheadattention models (default=8).
        max_len: The max sequence length (default=5000).
        num_decoder_layers: The number of decoder layers (default=6).
        dim_feedforward: The dimension of the feedforward network (default=2048).
        dropout: The dropout value (default=0.1).
        activation: The activation function (default: relu).
        layer_norm_eps: The eps value in layer normalization (default=1e-5).
        batch_first: If True, input is (batch, seq, feature) (default: True).
        norm_first: If True, layer norm is done before attention (default: False).
        bias: If True, Linear and LayerNorm learn bias (default: True).
        device: The device to use.
        dtype: The data type to use.
    """

    def __init__(
        self,
        n_vocab: int = 30000,
        d_model: int = 512,
        nhead: int = 8,
        max_len: int = 5000,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.input_emb = nn.Embedding(n_vocab, d_model, **factory_kwargs)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len).to(device)

        # Load the FLAN-T5 encoder
        self.encoder = T5EncoderModel.from_pretrained("google/flan-t5-base").to(device)
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            **factory_kwargs,
        )
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.projection = nn.Linear(d_model, n_vocab).to(device)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor,
        tgt: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = True,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """Forward pass through the transformer.

        Args:
            src: The source sequence (input text token IDs).
            src_mask: The attention mask for the source sequence.
            tgt: The target sequence (current MIDI tokens).
            memory_mask: The mask for the encoder output.
            memory_key_padding_mask: The mask for memory keys per batch.
            tgt_is_causal: If True, apply causal mask for target.
            memory_is_causal: If True, apply causal mask for memory.

        Returns:
            Logits for next token prediction.
        """
        if src.dim() != tgt.dim():
            raise RuntimeError("the number of dimensions in src and tgt must be equal")

        # Encode source using T5 encoder
        memory = self.encoder(src, attention_mask=src_mask).last_hidden_state

        # Embed target tokens
        tgt = self.input_emb(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        # Decode
        output = self.decoder(
            tgt,
            memory,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )

        # Project to vocabulary
        output = self.projection(output)

        return output

    def generate(
        self,
        src: Tensor,
        src_mask: Tensor,
        max_len: int = 100,
        temperature: float = 1.0,
        tgt_fin: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate MIDI tokens from text input.

        Args:
            src: The source sequence (input text token IDs).
            src_mask: The attention mask for the source sequence.
            max_len: The maximum number of tokens to generate (default=100).
            temperature: The temperature for sampling (default=1.0).
            tgt_fin: Optional starting tokens for progressive generation.

        Returns:
            Generated token sequence (only the NEW tokens if tgt_fin is provided, 
            or all tokens excluding the start token if tgt_fin is None).
        """
        if src.dim() != 2:
            raise RuntimeError("The src tensor should be 2-dimensional")

        # Initialize with start token
        if tgt_fin is None:
            # Token ID 1 is used in legacy code when starting fresh
            tgt_fin = torch.full((src.size(0), 1), 1, dtype=torch.long, device=src.device)
        else:
            # When continuing, we must prepend BOS token (1) to maintain context
            if isinstance(tgt_fin, list):
                tgt_fin = [1] + tgt_fin
            else:
                tgt_fin = [1] + tgt_fin.tolist()
            value_tensor = torch.tensor(tgt_fin, dtype=torch.long, device=src.device)
            tgt_fin = value_tensor.unsqueeze(0).repeat(src.size(0), 1)

        original_len = tgt_fin.size(1)

        for _ in range(max_len):
            tgt = tgt_fin
            output = self.forward(
                src,
                src_mask,
                tgt,
                memory_mask=None,
                memory_key_padding_mask=None,
                tgt_is_causal=True,
                memory_is_causal=False,
            )

            # Get logits for the last position
            logits = output[:, -1, :]

            # Apply temperature and sample
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tgt_fin = torch.cat((tgt_fin, next_token), dim=1)

        # Return only the newly generated tokens
        return tgt_fin[:, original_len:]

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Generate a square causal mask for the sequence."""
        return _generate_square_subsequent_mask(sz, dtype=dtype, device=device)

    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
