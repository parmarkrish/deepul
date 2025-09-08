import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, attn_mask=None, is_causal=False):
        super().__init__()
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.is_causal = is_causal
        self.attn_mask = attn_mask
        self.num_heads = num_heads
        self.scaling_factor = math.sqrt(d_model / num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, S, D = query.shape
        B, T, D = value.shape
        H = self.num_heads

        # (B, S, D) @ (D, D) -> (B, S, D) -> (B, S, H, D/H) -> (B, H, S, D/H)
        Q = self.query(query).view(B, S, H, D//H).transpose(1, 2)
        # (B, T, D) @ (D, D) -> (B, T, D) -> (B, T, H, D/H) -> (B, H, T, D/H)
        K = self.key(key).view(B, T, H, D//H).transpose(1, 2)
        # (B, T, D) @ (D, D) -> (B, T, D) -> (B, T, H, D/H) -> (B, H, T, D/H)
        V = self.value(value).view(B, T, H, D//H).transpose(1, 2)

        # (B, H, S, D/H) @ (B, H, D/H, T) -> (B, H, S, T)
        Y = torch.matmul(Q, K.transpose(2, 3)) / self.scaling_factor

        if self.is_causal:
            self.attn_mask = torch.tril(torch.ones(S, T))

        # We apply value -inf so that softmax for that element equals 0
        if self.attn_mask is not None:
            Y = Y.masked_fill(self.attn_mask == 0, float("-inf"))

        # (B, H, S, T) @ (B, H, T, D/H) -> (B, H, S, D/H)
        Y = self.dropout(F.softmax(Y, dim=-1)) @ V
        # (B, S, H, D/H) -> (B, S, D) @ (D, D) -> (B, S, D)
        out = self.proj(Y.transpose(1, 2).reshape(B, S, D))
        return out


class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self, x):
        return super().forward(x, x, x)


class Block(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_mlp=2048, dropout=0.1, act=nn.GELU(), attn_mask=None, is_causal=True, use_kv_cache=False):
        super().__init__()
        Attn = MultiHeadAttentionWithKVCache if use_kv_cache else MultiHeadAttention
        self.causal_attn = Attn(d_model, n_heads, dropout=dropout, attn_mask=attn_mask, is_causal=is_causal)
        self.msa = lambda x: self.causal_attn(x, x, x)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), act, nn.Linear(d_mlp, d_model))

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        out = x + self.mlp(self.ln2(x))
        return out


class Transformer(nn.Module):
    def __init__(self, n_tokens, max_seq_len, n_layers=2, d_model=128, n_heads=4, d_mlp=2048, dropout=0.1, act=nn.GELU(), attn_mask=None, is_causal=True):
        super().__init__()
        self.n_tokens = n_tokens
        self.max_seq_len = max_seq_len

        self.emb = nn.Embedding(n_tokens, d_model)
        self.emb.weight.data *= 0.1  # small weight initialization

        self.pos_enc = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.1)

        self.blocks = nn.Sequential(*[
            Block(d_model, n_heads, d_mlp, dropout, act, attn_mask, is_causal) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_tokens-1)  # we don't want net to predict <bos>; hence the -1

    def forward(self, x):
        _, seq_len = x.shape

        x = self.emb(x) + self.pos_enc[:seq_len, :] # (B, T, D)
        x = self.blocks(x)
        logits = self.head(self.ln(x))
        return logits

    def loss(self, y):  # (B, T)
        # an alternate method is having both x and y. 
        # But I just have y and prepend <bos> and remove last token to get x

        B, _ = y.shape
        x = torch.cat((torch.tensor(self.n_tokens-1).expand(B, 1), y[:, :-1]), dim=-1)

        logits = self.forward(x)
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))

    @torch.no_grad()
    def generate(self, n_samples: int):
        x = torch.zeros((n_samples, self.max_seq_len), dtype=torch.long)
        x[:, 0] = self.n_tokens-1 # add <bos> token
        for i in range(self.max_seq_len):
            logits = self.forward(x)  # (B, T, n_token-1)
            prob_dist = torch.softmax(logits[:, i, :], dim=-1)  # (B, n_token-1)
            samples = torch.multinomial(prob_dist, num_samples=1)  # (B, 1)

            if i == self.max_seq_len-1:
                return torch.cat((x[:, 1:], samples), dim=1)

            x[:, i+1] = samples.view(-1)


class MultiHeadAttentionWithKVCache(MultiHeadAttention):
    def __init__(self, d_model, num_heads, dropout=0.1, attn_mask=None, is_causal=False):
        super().__init__(d_model, num_heads, dropout, attn_mask, is_causal)
        self.KV_cache = None
        
    def forward_kv_cache(self, query, key, value):
        N, D = query.shape
        H = self.num_heads
        # (N, D) @ (D, D) -> (N, D) -> (N, H, D//H)
        Q = self.query(query).view(N, H, 1, D//H)
        K = self.key(key).view(N, H, D//H, 1)
        V = self.value(value).view(N, H, 1, D//H)

        if self.KV_cache is not None:  # append to KV cache
            self.KV_cache['keys'] = torch.cat([self.KV_cache['keys'], K.view(N, H, D//H, 1)], dim=-1)
            self.KV_cache['values'] = torch.cat([self.KV_cache['values'], V.view(N, H, 1, D//H)], dim=-2)
        else:  # init KV cache
            self.KV_cache = {'keys': K, 'values': V}  # keys: (N, H, D//H, S); values: (N, H, S, D//H)

        Y = (Q @ self.KV_cache['keys']) / self.scaling_factor   # (N, H, 1, D//H) @ (N, H, D//H, S) -> (N, H, 1, S)

        Y = self.dropout(F.softmax(Y, dim=-1)) @ self.KV_cache['values']  # (N, H, 1, S) @ (N, H, S, D//H)
        out = self.proj(Y.view(N, D))
        return out
    
    def forward(self, query, key, value):
        if self.training:
            return super().forward(query, key, value)
        return self.forward_kv_cache(query, key, value)


class TransformerWithKVCache(Transformer):
    def __init__(self, n_tokens, max_seq_len, n_layers=2, d_model=128, n_heads=4, d_mlp=2048, dropout=0.1, act=nn.GELU(), attn_mask=None, is_causal=True):
        super().__init__(n_tokens, max_seq_len, n_layers, d_model, n_heads, d_mlp, dropout, act, attn_mask, is_causal)
        self.pos = 0
        self.blocks = nn.Sequential(*[
            Block(d_model, n_heads, d_mlp, dropout, act, attn_mask, is_causal, use_kv_cache=True) for _ in range(n_layers)
        ])

    def forward(self, x):
        if self.training:  # x: (B, S)
            _, seq_len = x.shape
            pos_enc = self.pos_enc[:seq_len, :] # (B, S, D)
        else:  # x: (B, ) (we use KV cache in this case)
            pos_enc = self.pos_enc[self.pos, :]
            self.pos = min(self.pos + 1, self.max_seq_len - 1)

        x = self.emb(x) + pos_enc
        x = self.blocks(x)
        logits = self.head(self.ln(x))
        return logits

    @torch.no_grad()
    def generate(self, n_samples: int, use_KV_cache=True):
        self.eval() if use_KV_cache else self.train()

        x = torch.zeros((n_samples, self.max_seq_len), dtype=torch.long)
        x[:, 0] = self.n_tokens-1 # add <bos> token

        start_event = [torch.cuda.Event(enable_timing=True) for i in range(self.max_seq_len)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(self.max_seq_len)]
        step_times_ms = []

        for i in range(self.max_seq_len):
            start_event[i].record()

            logits = self.forward(x[:, i] if use_KV_cache else x)
            prob_dist = torch.softmax(logits if use_KV_cache else logits[:, i, :], dim=-1)  # (N, n_token-1)
            samples = torch.multinomial(prob_dist, num_samples=1)  # (N, 1)

            if i == self.max_seq_len - 1:
                x = torch.cat((x[:, 1:], samples), dim=1)
            else:
                x[:, i + 1] = samples.view(-1)
    
            end_event[i].record()
            torch.cuda.synchronize()
            step_times_ms.append(start_event[i].elapsed_time(end_event[i]))

        return step_times_ms, x

# transformer with loss that takes in both inputs and targets
class TransformerXY(Transformer):
    def __init__(self, n_tokens, max_seq_len, n_layers=2, d_model=128, n_heads=4, d_mlp=2048, dropout=0.1, act=nn.GELU(), attn_mask=None, is_causal=True):
        super().__init__(n_tokens, max_seq_len, n_layers, d_model, n_heads, d_mlp, dropout, act, attn_mask, is_causal)
        self.head = nn.Linear(d_model, n_tokens)

    def loss(self, x, y):  # (B, S)
        logits = self.forward(x)  # (B, S, T)
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), ignore_index=-1)
    
    @torch.no_grad()
    def generate(self, n_samples: int, bos_token=None):
        x = torch.zeros((n_samples, self.max_seq_len), dtype=torch.long)
        x[:, 0] = 0  # add bos_token
        for i in range(self.max_seq_len):
            logits = self.forward(x)
            prob_dist = torch.softmax(logits[:, i, :], dim=-1)  # (N, n_token-1)
            samples = torch.multinomial(prob_dist, num_samples=1)  # (N, 1)

            if i == self.max_seq_len - 1:
                return torch.cat((x[:, 1:], samples), dim=1)

            x[:, i+1] = samples.view(-1)
