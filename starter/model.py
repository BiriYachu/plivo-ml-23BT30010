import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    vocab_size = 256
    block_size = 128
    n_layer = 4
    n_head = 6
    n_embd = 192
    dropout = 0.0
    tie_weights = True

def rotate_half(x):
    x_rot = torch.zeros_like(x)
    x_rot[..., 0::2] = -x[..., 1::2]
    x_rot[..., 1::2] = x[..., 0::2]
    return x_rot

def apply_rope(x, cos, sin):
    cos_t = cos[:x.shape[1], None, :]
    sin_t = sin[:x.shape[1], None, :]
    return x * cos_t + rotate_half(x) * sin_t

class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = apply_rope(q, cos, sin).transpose(1, 2)
        k = apply_rope(k, cos, sin).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
            
        head_dim = cfg.n_embd // cfg.n_head
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(cfg.block_size, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())
        
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos = self.cos[:T]
        sin = self.sin[:T]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
