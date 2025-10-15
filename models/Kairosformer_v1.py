# Kairosformer_v1.py
from typing import Optional
import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_wo_pos
from layers.Autoformer_EncDec import Decoder, DecoderLayer, my_Layernorm, series_decomp
from layers.Transformer_EncDec import ConvLayer, Encoder as Transformer_Encoder, EncoderLayer as Transformer_EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, ProbAttention
from layers.AutoCorrelation import AutoCorrelation
from layers.HybridAttention import HybridAttention


def make_inner_attention(kind: str, mask_flag: bool, configs):
    """
    Factory for inner attention module used by AttentionLayer.
    kind ∈ {'prob', 'auto', 'hybrid'}
    All return a module with signature:
      (Q,K,V,mask) -> (out, attn)
    where Q/K/V shapes are [B, L, H, d].
    """
    kind = (kind or 'hybrid').lower()
    if kind == 'prob':
        return ProbAttention(
            mask_flag=mask_flag,
            factor=getattr(configs, 'factor', 1),
            attention_dropout=configs.dropout,
            output_attention=False,
        )
    elif kind == 'auto':
        return AutoCorrelation(
            mask_flag=mask_flag,
            factor=getattr(configs, 'factor', 1),
            attention_dropout=configs.dropout,
            output_attention=False,
        )
    elif kind == 'hybrid':
        return HybridAttention(
            mask_flag=mask_flag,
            factor=getattr(configs, 'factor', 1),
            attention_dropout=configs.dropout,
            output_attention=False,
            gate_temp=getattr(configs, 'gate_temp', 1.0),
        )
    else:
        raise ValueError(f"Unknown attention kind: {kind}")


def make_attention_layer(kind: str, mask_flag: bool, configs):
    """Wrap inner attention with projections to/from [B,L,D]."""
    return AttentionLayer(
        make_inner_attention(kind, mask_flag, configs),
        d_model=configs.d_model,
        n_heads=configs.n_heads,
    )


class Model(nn.Module):
    """
    Kairosformer v1
    - Encoder: Transformer_EncoderLayer with configurable attention (prob/auto/hybrid)
      plus Informer-style ConvLayer downsampling, and Autoformer-style my_Layernorm.
    - Decoder: Autoformer progressive decomposition Decoder, with configurable attention.
    - Input is decomposed to seasonal/trend; seasonal to encoder; trend initializes decoder trend branch.
    - Switch attention via configs.attn ∈ {'prob','auto','hybrid'} (default 'hybrid').
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len   = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len  = configs.pred_len
        self.attn_kind = getattr(configs, 'attn', 'hybrid').lower()

        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decomp = series_decomp(kernel_size=kernel_size)

        # embeddings (no absolute positional embeddings; using time features + token conv)
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # --- Encoder ---
        enc_layers = []
        for _ in range(configs.e_layers):
            enc_layers.append(
                Transformer_EncoderLayer(
                    attention=make_attention_layer(self.attn_kind, mask_flag=False, configs=configs),
                    d_model=configs.d_model,
                    d_ff=min(configs.d_ff, int(2.5 * configs.d_model)),
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
            )

        self.encoder = Transformer_Encoder(
            attn_layers=enc_layers,
            conv_layers=[ConvLayer(configs.d_model) for _ in range(configs.e_layers - 1)],
            norm_layer=my_Layernorm(configs.d_model),
        )

        # --- Decoder ---
        def _self_attn():
            return make_attention_layer(self.attn_kind, mask_flag=True,  configs=configs)

        def _cross_attn():
            return make_attention_layer(self.attn_kind, mask_flag=False, configs=configs)

        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    self_attention=_self_attn(),
                    cross_attention=_cross_attn(),
                    d_model=configs.d_model,
                    c_out=configs.c_out,
                    d_ff=configs.d_ff,
                    moving_avg=kernel_size,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
        dec_enc_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_enc: [B, L_in, C]
        x_dec: (ignored; we build decoder seasonal/trend inits ourselves)
        returns: [B, pred_len, C]
        """
        B, L, C = x_enc.shape

        # series decomposition
        seasonal_all, trend_all = self.decomp(x_enc)  # both [B, L, C]

        # decoder inits
        mean  = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.pred_len, 1)       # [B, pred, C]
        zeros = torch.zeros((B, self.pred_len, C), device=x_enc.device, dtype=x_enc.dtype)

        trend_init    = torch.cat([trend_all[:,   -self.label_len:, :], mean],  dim=1)   # [B, label+pred, C]
        seasonal_init = torch.cat([seasonal_all[:, -self.label_len:, :], zeros], dim=1)  # [B, label+pred, C]

        # encoder: we feed seasonal component only (as in Autoformer)
        enc_in  = seasonal_all
        enc_out = self.enc_embedding(enc_in, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)  # [B, L, D]

        # decoder
        dec_in = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            x=dec_in,
            cross=enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init,
        )

        out = seasonal_part + trend_part  # [B, label+pred, C]
        return out[:, -self.pred_len:, :]
