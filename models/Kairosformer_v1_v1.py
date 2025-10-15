# models/Kairosformer_v1.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Simple Embeddings (value + pos)
# -------------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x):  # x: [B, L, D]
        L = x.size(1)
        return self.pe[:L].unsqueeze(0)  # [1, L, D]


class DataEmbeddingSimple(nn.Module):
    """
    Minimal value embedding + positional embedding.
    Expects inputs shaped [B, L, C] -> projects to [B, L, d_model].
    """
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_proj = nn.Linear(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, L, C]
        v = self.value_proj(x)
        p = self.pos_emb(v)
        return self.drop(v + p)
    


# -----------------------------------------
# Two per-head kernels with identical I/O
# I/O contract: Q,K,V: [B, L*, H, Dh] -> out: [B, Lq, H, Dh]
# -----------------------------------------
# --- ProbSparse (Informer) ---
class ProbSparseHead(nn.Module):
    def __init__(self, factor=5, dropout=0.1, **_):
        super().__init__()
        self.factor = factor          # controls top-u queries: u = factor * ln(Lk)
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        B, Lq, H, Dh = Q.shape
        # in ProbSparseHead.forward
        Lk = K.size(1); Lq = Q.size(1)
        if Lk <= 128:            # or 256 — pick a threshold
            # fall back to dense dot-product attention (still [B,L,H,Dh] I/O)
            attn = torch.einsum('blhd,bs hd->blhs', Q, K) / math.sqrt(Dh)  # [B,Lq,H,Lk]
            P = torch.softmax(attn, dim=-1)
            out = torch.einsum('blhs,bshd->blhd', P, V)
            return self.drop(out), None
        u = max(1, int(self.factor * math.log(max(Lk, 2))))     # Informer rule
        m = min(64, Lk)                                         # sample size
        idx = torch.randint(0, Lk, (m,), device=Q.device)
        Ks = K[:, idx]                                          # [B,m,H,Dh]
        # simple sparsity score proxy (use your preferred one)
        score = (Q * Ks.mean(dim=1, keepdim=True)).sum(-1).abs().mean(2)  # [B,Lq]
        topu = score.topk(u, dim=1).indices                     # [B,u]
        Q_top = Q.gather(1, topu[..., None, None].expand(-1,-1,H,Dh))
        attn = torch.einsum('buhd,blhd->buhl', Q_top, K) / math.sqrt(Dh)
        if mask is not None: attn = attn.masked_fill(mask, float('-inf'))
        P = torch.softmax(attn, dim=-1)
        Y_top = torch.einsum('buhl,blhd->buhd', P, V)           # [B,u,H,Dh]
        ctx = V.mean(dim=1, keepdim=True).expand(B, Lq, H, Dh)
        out = ctx.clone()
        out.scatter_(1, topu[..., None, None].expand_as(Y_top), Y_top)
        return self.drop(out), None

# --- AutoCorrelation (Autoformer) ---
class AutoCorrelationHead(nn.Module):
    def __init__(self, k_delays=4, dropout=0.1, **_):
        super().__init__()
        self.k_delays = k_delays
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # length-safe: make K,V match Lq
        B, Lq, H, Dh = Q.shape
        S = V.size(1)
        if Lq > S:
            pad = torch.zeros(B, Lq - S, H, Dh, device=V.device, dtype=V.dtype)
            K = torch.cat([K, pad], dim=1); V = torch.cat([V, pad], dim=1)
        elif Lq < S:
            K = K[:, :Lq]; V = V[:, :Lq]

        qf = torch.fft.rfft(Q.permute(0,2,3,1), dim=-1)
        kf = torch.fft.rfft(K.permute(0,2,3,1), dim=-1)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=Lq, dim=-1)   # [B,H,Dh,Lq]

        score = corr.mean(dim=(1,2))                                # [B,Lq]
        delays = score.topk(self.k_delays, dim=-1).indices          # [B,k]
        weights = torch.softmax(score.gather(1, delays), dim=-1)    # [B,k]

        out = 0
        for i in range(self.k_delays):
            d = delays[:, i]
            Vd = torch.stack([torch.roll(V[b], shifts=int(d[b]), dims=0) for b in range(B)], dim=0)
            out = out + weights[:, i].view(B,1,1,1) * Vd
        return self.drop(out), None

# ------------------------------------------------
# HybridAttention: one QKV, packed head groups
# ------------------------------------------------
class HybridAttention(nn.Module):
    """
    - One QKV projection
    - Head groups executed per-kind without per-head branching
    - Returns [B, Lq, d_model]
    """
    def __init__(self, d_model, n_heads, dropout=0.1,
                 head_plan=("auto", 8, "prob", 0)):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads

        # Build static head plan → contiguous spans: [("auto",0,2), ("prob",2,8)]
        kinds = head_plan[::2]
        counts = head_plan[1::2]
        assert sum(counts) == n_heads, "sum(head counts) must equal n_heads"
        self.groups = []
        h0 = 0
        for k, c in zip(kinds, counts):
            assert k in ("auto", "prob"), f"unknown head kind {k}"
            self.groups.append((k, h0, h0 + c))
            h0 += c

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_drop = nn.Dropout(dropout)

        # Kernels
        self.auto_attn = AutoCorrelationHead(dropout=dropout)
        self.prob_attn = ProbSparseHead(dropout=dropout)

    def _run_group(self, kind, Q, K, V, mask):
        if kind == "auto":
            y, _ = self.auto_attn(Q, K, V, mask)
        else:
            y, _ = self.prob_attn(Q, K, V, mask)
        return y  # [B,Lq,Hg,Dh]

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries: [B, Lq, d_model]
        keys:    [B, Lk, d_model]
        values:  [B, Lk, d_model]
        returns: [B, Lq, d_model]
        """
        B, Lq, D = queries.shape
        _, Lk, _ = keys.shape
        H, Dh = self.n_heads, self.dh

        qkv_q = self.qkv(queries)
        qkv_k = self.qkv(keys)
        qkv_v = self.qkv(values)
        Q = qkv_q[..., :D].view(B, Lq, H, Dh).contiguous()
        K = qkv_k[..., D:2*D].view(B, Lk, H, Dh).contiguous()
        V = qkv_v[..., 2*D:].view(B, Lk, H, Dh).contiguous()

        outs = []
        for kind, h0, h1 in self.groups:
            Qi = Q[:, :, h0:h1, :]
            Ki = K[:, :, h0:h1, :]
            Vi = V[:, :, h0:h1, :]
            yi = self._run_group(kind, Qi, Ki, Vi, attn_mask)  # [B,Lq,Hg,Dh]
            outs.append(yi)

        Y = torch.cat(outs, dim=2)             # [B,Lq,H,Dh]
        Y = Y.view(B, Lq, H * Dh).contiguous() # [B,Lq,D]
        Y = self.o_proj(self.out_drop(Y))
        return Y


# -----------------------------------------
# Minimal Kairosformer v1 "Model" wrapper
# -----------------------------------------
class Model(nn.Module):
    """
    Single-file Kairosformer v1:
      - Simple value+pos embeddings for encoder/decoder
      - HybridAttention for decoder self-attn and cross-attn
      - Final projection to c_out
    It matches the Exp API: forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
    and returns [B, L_pred, C_out].
    """
    def __init__(self, args):
        super().__init__()
        # Required args (fall back to sensible defaults if missing)
        d_model  = getattr(args, "d_model", 512)
        n_heads  = getattr(args, "n_heads", 8)
        dropout  = getattr(args, "dropout", 0.1)
        enc_in   = getattr(args, "enc_in", 7)
        dec_in   = getattr(args, "dec_in", enc_in)
        c_out    = getattr(args, "c_out", enc_in)
        # seq/label/pred lengths (used to slice outputs)
        self.pred_len = getattr(args, "pred_len", 96)

        # Head plan: tweak ratios here (e.g., 25% auto, 75% prob)
        auto_heads = max(1, n_heads // 4)
        prob_heads = n_heads - auto_heads
        head_plan = ("auto", 8, "prob", 0)

        # Embeddings
        self.enc_emb = DataEmbeddingSimple(enc_in, d_model, dropout=dropout)
        self.dec_emb = DataEmbeddingSimple(dec_in, d_model, dropout=dropout)

        # Attention blocks
        self.self_attn  = HybridAttention(d_model, n_heads, dropout=dropout, head_plan=head_plan)
        self.cross_attn = HybridAttention(d_model, n_heads, dropout=dropout, head_plan=head_plan)

        # Simple FFN (Transformer-style)
        d_ff = getattr(args, "d_ff", 4 * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Output head
        self.proj = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc:      [B, L_enc, C_in]
        x_mark_enc: [B, L_enc, *] (ignored here)
        x_dec:      [B, L_dec, C_in]
        x_mark_dec: [B, L_dec, *] (ignored here)
        returns:    [B, L_dec_tail(self.pred_len), C_out]
        """
        B, L_enc, _ = x_enc.shape
        _, L_dec, _ = x_dec.shape

        # Embeddings
        enc = self.enc_emb(x_enc)  # [B,L_enc,D]
        dec = self.dec_emb(x_dec)  # [B,L_dec,D]

        # Decoder self-attention (pre-norm)
        y = self.norm1(dec)
        y = self.self_attn(y, y, y, attn_mask=None) + dec

        # Cross-attention with encoder memory
        z = self.norm2(y)
        z = self.cross_attn(z, enc, enc, attn_mask=None) + y

        # FFN
        o = self.norm3(z)
        o = self.ffn(o) + z  # [B,L_dec,D]

        # Project to outputs
        out = self.proj(o)   # [B,L_dec,C_out]

        # Return only the last pred_len steps from decoder
        if self.pred_len is not None and self.pred_len > 0 and self.pred_len <= out.size(1):
            out = out[:, -self.pred_len:, :]
        return out
