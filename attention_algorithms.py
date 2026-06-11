import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
    """Build a causal mask for attention scores."""
    query_positions = torch.arange(query_len, device=device).unsqueeze(-1)
    key_positions = torch.arange(key_len, device=device).unsqueeze(0)
    return key_positions > query_positions


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Standard scaled dot-product attention."""
    # q, k, v shape: [B, H, T, D]
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        mask = causal_mask(q.size(2), k.size(2), q.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def block_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    causal: bool = False,
) -> torch.Tensor:
    """Block-wise attention in pure PyTorch as a simplified FlashAttention-style kernel.

    This processes query slices in blocks to reduce peak memory usage compared to materializing a
    full [T, T] score matrix for the entire sequence.
    """
    B, H, T, D = q.shape
    outputs = []
    scale = 1.0 / math.sqrt(D)

    for start in range(0, T, block_size):
        end = min(start + block_size, T)
        q_block = q[:, :, start:end, :]
        scores = torch.matmul(q_block, k.transpose(-2, -1)) * scale

        if causal:
            mask = causal_mask(end - start, T, q.device)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        outputs.append(torch.matmul(weights, v))

    return torch.cat(outputs, dim=2)


def lean_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """A simple linearized LeanAttention approximation in pure PyTorch."""
    # Feature map that keeps values positive and avoids softmax
    q_feat = F.elu(q) + 1.0
    k_feat = F.elu(k) + 1.0

    if causal:
        kv_prefix = torch.cumsum(k_feat * v, dim=2)
        k_prefix = torch.cumsum(k_feat, dim=2)
        normalizer = (q_feat * k_prefix).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return q_feat * kv_prefix / normalizer

    kv = torch.einsum("...nd,...ne->...nde", k_feat, v).sum(dim=2)
    normalizer = torch.einsum("...nd,...nd->...n", q_feat, k_feat).unsqueeze(-1).clamp(min=1e-6)
    return torch.matmul(q_feat, kv) / normalizer


class FullAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must divide evenly into num_heads"
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attended = scaled_dot_product_attention(q, k, v, causal=causal)
        attended = attended.transpose(1, 2).reshape(B, T, C)
        return self.out(attended)


class BlockFlashAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must divide evenly into num_heads"
        self.block_size = block_size
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attended = block_flash_attention(q, k, v, self.block_size, causal=causal)
        attended = attended.transpose(1, 2).reshape(B, T, C)
        return self.out(attended)


class FlashDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must divide evenly into num_heads"
        self.block_size = block_size
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward_step(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if k_cache is not None and v_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        attended = block_flash_attention(q, k, v, self.block_size, causal=True)
        attended = attended.transpose(1, 2).reshape(B, T, C)
        out = self.out(attended)
        return out, k, v


class LeanAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must divide evenly into num_heads"
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attended = lean_attention(q, k, v, causal=causal)
        attended = attended.transpose(1, 2).reshape(B, T, C)
        return self.out(attended)


def example_usage() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    seq_len = 128
    embed_dim = 128
    num_heads = 8

    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    full = FullAttention(embed_dim, num_heads).to(device)
    block = BlockFlashAttention(embed_dim, num_heads, block_size=32).to(device)
    lean = LeanAttention(embed_dim, num_heads).to(device)
    decoder = FlashDecoder(embed_dim, num_heads, block_size=32).to(device)

    out_full = full(x, causal=True)
    out_block = block(x, causal=True)
    out_lean = lean(x, causal=True)

    print("Full attention output shape:", out_full.shape)
    print("Block flash attention output shape:", out_block.shape)
    print("Lean attention output shape:", out_lean.shape)

    step_input = x[:, :1, :]
    out_step, k_cache, v_cache = decoder.forward_step(step_input)
    print("Decoder step output shape:", out_step.shape)
    print("Cache K shape:", k_cache.shape, "Cache V shape:", v_cache.shape)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_model(
    model: nn.Module,
    x: torch.Tensor,
    iterations: int = 20,
    warmup: int = 5,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        _synchronize(x.device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(x)
        _synchronize(x.device)
    return (time.perf_counter() - start) / iterations


def benchmark_decoder(
    decoder: FlashDecoder,
    x_step: torch.Tensor,
    iterations: int = 20,
    warmup: int = 5,
) -> float:
    decoder.eval()
    with torch.no_grad():
        k_cache = None
        v_cache = None
        for _ in range(warmup):
            _, k_cache, v_cache = decoder.forward_step(x_step, k_cache, v_cache)
        _synchronize(x_step.device)
        start = time.perf_counter()
        for _ in range(iterations):
            _, k_cache, v_cache = decoder.forward_step(x_step, k_cache, v_cache)
        _synchronize(x_step.device)
    return (time.perf_counter() - start) / iterations


def benchmark_attention() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    seq_len = 256
    embed_dim = 128
    num_heads = 8

    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    x_step = torch.randn(batch_size, 1, embed_dim, device=device)

    full = FullAttention(embed_dim, num_heads).to(device)
    block = BlockFlashAttention(embed_dim, num_heads, block_size=32).to(device)
    lean = LeanAttention(embed_dim, num_heads).to(device)
    decoder = FlashDecoder(embed_dim, num_heads, block_size=32).to(device)

    full_time = benchmark_model(full, x)
    block_time = benchmark_model(block, x)
    lean_time = benchmark_model(lean, x)
    decoder_time = benchmark_decoder(decoder, x_step)

    print("\nBenchmark results (average time per forward pass):")
    print(f"Full attention:   {full_time * 1000:.2f} ms")
    print(f"Block flash:      {block_time * 1000:.2f} ms")
    print(f"Lean attention:   {lean_time * 1000:.2f} ms")
    print(f"Decoder step:     {decoder_time * 1000:.2f} ms")


if __name__ == "__main__":
    example_usage()
    benchmark_attention()
