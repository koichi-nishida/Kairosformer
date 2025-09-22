from typing import Optional
import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Decoder, DecoderLayer, my_Layernorm, series_decomp
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Transformer_EncDec import ConvLayer, Encoder as Transformer_Encoder, EncoderLayer as Transformer_EncoderLayer


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size=getattr(configs, 'moving_avg', 25))

        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        self.encoder = Transformer_Encoder(
            [
                Transformer_EncoderLayer(
                    AutoCorrelationLayer(
                        ProbAttention(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                    ),
                    d_model=configs.d_model,
                    d_ff= min(configs.d_ff, int(2.5 * configs.d_model)),
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            conv_layers=[ConvLayer(configs.d_model) for _ in range(configs.e_layers - 1)],
            norm_layer=my_Layernorm(configs.d_model),
        )

        def _self_attn():
            return AttentionLayer(
                ProbAttention(
                    mask_flag=True,
                    factor=getattr(configs, 'factor', 1),
                    attention_dropout=configs.dropout,
                    output_attention=False,
                ),
                d_model=configs.d_model,
                n_heads=configs.n_heads,
            )

        def _cross_attn():
            return AttentionLayer(
                ProbAttention(
                    mask_flag=False,
                    factor=configs.factor,
                    attention_dropout=configs.dropout,
                    output_attention=False,
                ),
                d_model=configs.d_model,
                n_heads=configs.n_heads,
            )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    self_attention=_self_attn(),
                    cross_attention=_cross_attn(),
                    d_model=configs.d_model,
                    c_out=configs.c_out,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
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
        B, _, C = x_enc.shape
        seasonal_all, trend_all = self.decomp(x_enc)
        mean  = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.pred_len, 1)        
        zeros = torch.zeros((B, self.pred_len, C), device=x_enc.device, dtype=x_enc.dtype)  

        trend_init    = torch.cat([trend_all[:,   -self.label_len:, :], mean],  dim=1)     
        seasonal_init = torch.cat([seasonal_all[:, -self.label_len:, :], zeros], dim=1)   

        enc_in = seasonal_all                            
        enc_out = self.enc_embedding(enc_in, x_mark_enc)   
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_in = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_in,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init,
        )

        dec_out = seasonal_part + trend_part
        out = dec_out[:, -self.pred_len :, :]
        return out