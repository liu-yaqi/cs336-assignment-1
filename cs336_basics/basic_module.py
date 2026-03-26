import torch
import torch.nn as nn
from typing import Any


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = torch.max(x, dim=dim, keepdim=True).values
    x = x - max_x
    exp_x = torch.exp(x)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    sum_exp = torch.clamp(sum_exp, min=1e-10)
    return exp_x / sum_exp


class Linear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) / input_dim**0.5 ) # note: kaiming init

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return in_features @ self.weight.T
    

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embedding_dim)) # note: kaiming init

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(SwiGLU, self).__init__()
        self.w1 = Linear(d_model, d_ff) # d_ff * 2
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w3(x)
        return self.w2(silu(x1) * x2)
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        d_k = Q.size()[-1]
        scores = (Q @ K.transpose(-2, -1)) * (d_k**(-0.5))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        output = attention_weights @ V
        return output
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_seq, _ = x.size()
        Q = self.W_q(x).view(batch, n_seq, self.num_heads, self.d_k).transpose(-2,-3)
        K = self.W_k(x).view(batch, n_seq, self.num_heads, self.d_k).transpose(-2,-3)
        V = self.W_v(x).view(batch, n_seq, self.num_heads, self.d_k).transpose(-2,-3)
        mask = torch.tril(torch.ones(n_seq, n_seq))

        output = self.attention(Q, K, V, mask).transpose(-2,-3).contiguous().view(batch, n_seq, self.num_heads * self.d_k)
        output = self.W_o(output)  
        return output


class RoPE(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 2048, theta_base: float = 10000.0):
        super(RoPE, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(theta_base)) / d_model))

        freq_cis = torch.polar(torch.ones(max_seq_len, d_model // 2), position * div_term) # shape: (max_seq_len, d_model/2)
        self.register_buffer('freq_cis', freq_cis)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_cmp = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe = self.freq_cis[token_positions, :]
        x_rotated = torch.view_as_real(x_cmp * pe).flatten(-2)
        return x_rotated


class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048, theta_base: float = 10000.0):
        super(MultiHeadSelfAttentionRoPE, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.rope = RoPE(self.d_k, max_seq_len, theta_base)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        batch, n_seq, _ = x.size()
        Q = self.q_proj(x).view(batch, n_seq, self.num_heads, self.d_k).transpose(-2,-3)
        K = self.k_proj(x).view(batch, n_seq, self.num_heads, self.d_k).transpose(-2,-3)
        V = self.v_proj(x).view(batch, n_seq, self.num_heads, self.d_k).transpose(-2,-3)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(n_seq, n_seq))

        output = self.attention(Q, K, V, mask).transpose(-2,-3).contiguous().view(batch, n_seq, self.num_heads * self.d_k)
        output = self.output_proj(output)  
        return output
    

class RmsNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super(RmsNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 2048, theta_base: float = 10000.0, eps: float = 1e-8):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttentionRoPE(d_model, num_heads, max_seq_len, theta_base)
        self.ln1 = RmsNorm(d_model, eps)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RmsNorm(d_model, eps)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.size(1), device=x.device)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_seq_len: int = 2048, theta_base: float = 10000.0, eps: float = 1e-8):
        super(TransformerLM, self).__init__()
        self.max_seq_len = max_seq_len
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta_base, eps)
            for _ in range(num_layers)
        ])
        self.ln_final = RmsNorm(d_model, eps)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward_old(self, inputs, targets):
        # inf
        probs = softmax(inputs, dim=-1)
        # print(probs)
        return torch.mean(-1.0 * torch.gather(torch.log(probs + 1e-10), dim=-1, index=targets.unsqueeze(-1)))
    
    def forward(self, inputs, targets):
        logit_target = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        max_x = inputs.max(dim=-1, keepdim=True).values
        log_sum_exp = max_x.squeeze(-1) + torch.log(torch.sum(torch.exp(inputs - max_x), dim=-1))
        return torch.mean(-1.0 * (logit_target - log_sum_exp))


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if not (0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    if temperature == 0:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits / temperature, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Keep the smallest prefix whose cumulative mass reaches top_p.
        sorted_remove_mask = cumulative_probs > top_p
        sorted_remove_mask[..., 1:] = sorted_remove_mask[..., :-1].clone()
        sorted_remove_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_remove_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        sampled_sorted_index = torch.multinomial(sorted_probs, num_samples=1)
        sampled_token = sorted_indices.gather(dim=-1, index=sampled_sorted_index)
        return int(sampled_token.item())

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_tokens(
    model: nn.Module,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token_id: int | None = None,
    device: str | torch.device | None = None,
) -> list[int]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if len(prompt_token_ids) == 0:
        raise ValueError("prompt_token_ids must be non-empty")

    if device is None:
        first_param = next(model.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")

    generated = list(prompt_token_ids)
    was_training = model.training
    model.eval()

    model_max_seq_len = getattr(model, "max_seq_len", None)

    for _ in range(max_new_tokens):
        model_input = generated
        if model_max_seq_len is not None and len(model_input) > model_max_seq_len:
            model_input = model_input[-model_max_seq_len:]

        input_tensor = torch.tensor(model_input, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(input_tensor)
        next_token_logits = logits[0, -1, :]
        next_token_id = _sample_next_token(next_token_logits, temperature=temperature, top_p=top_p)

        generated.append(next_token_id)
        if end_token_id is not None and next_token_id == end_token_id:
            break

    if was_training:
        model.train()

    return generated


@torch.no_grad()
def decode(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token: str = "<|endoftext|>",
    device: str | torch.device | None = None,
) -> str:
    prompt_token_ids = tokenizer.encode(prompt)
    end_token_id = None
    if hasattr(tokenizer, "special_tokens") and end_token in tokenizer.special_tokens:
        end_token_id = tokenizer.dstoi[end_token.encode("utf-8")]

    generated_ids = generate_tokens(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        end_token_id=end_token_id,
        device=device,
    )
    return tokenizer.decode(generated_ids)
