# memory.py
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


def _fnv1a_hash_ngram(windows: torch.Tensor, seed: int, prime: int) -> torch.Tensor:
    """
    windows: (B, L, n) int64
    returns: (B, L) int64 hash
    FNV-1a-ish rolling over n tokens. Uses int64 overflow (wraparound).
    """
    assert windows.dtype == torch.int64
    h = torch.full(windows.shape[:-1], seed, device=windows.device, dtype=torch.int64)
    # iterate over n dimension (small: 4/8/16), this loop is fine
    for k in range(windows.shape[-1]):
        h = h ^ windows[..., k]
        h = h * prime
    return h


def _to_int64_signed(x: int) -> int:
    x &= (1 << 64) - 1          # keep low 64 bits
    if x >= (1 << 63):
        x -= (1 << 64)          # convert to signed range
    return x


@dataclass(frozen=True)
class MemorySpec:
    ns: Sequence[int] = (8, 16)
    heads: int = 2
    table_size: int = 65536  # buckets per head per n
    dim: int = 64


class HashedNgramMemory(nn.Module):
    """
    Engram-ish hashed n-gram memory that produces a vocab-space bias.

    Given idx: (B, T) tokens (int64), for each n in ns:
      - form windows of last n tokens at each position t >= n-1
      - hash windows into [0, table_size)
      - gather embeddings from multiple hash heads
      - sum/avg embeddings across heads and n branches
      - project to vocab bias: (B, T, vocab_size)

    No non-bijective normalization is applied.
    """

    def __init__(self, vocab_size: int, spec: MemorySpec):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.ns = tuple(int(n) for n in spec.ns)
        self.heads = int(spec.heads)
        self.table_size = int(spec.table_size)
        self.dim = int(spec.dim)

        # Per (n, head) embedding table
        # tables[n_idx][head] -> nn.Embedding(table_size, dim)
        self.tables = nn.ModuleList()
        for _n in self.ns:
            head_tables = nn.ModuleList([nn.Embedding(self.table_size, self.dim) for _ in range(self.heads)])
            self.tables.append(head_tables)

        # Project aggregated memory embedding -> vocab bias
        self.proj = nn.Linear(self.dim, self.vocab_size, bias=False)

        # Initialize small so it doesn't dominate early
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
        for n_tables in self.tables:
            for emb in n_tables:
                nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        # Different seeds/primes per head (fixed constants)
        # primes should be odd and large-ish
        self._seeds = [_to_int64_signed(0xCBF29CE484222325 + 0x9E3779B97F4A7C15 * h)
                for h in range(self.heads)]
        self._primes = [_to_int64_signed(0x100000001B3 + 0x9E3779B97F4A7C15 * (h + 1))
                        for h in range(self.heads)]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) int64
        returns bias: (B, T, vocab_size) float (same dtype as embeddings/proj)
        """
        if idx.dtype != torch.int64:
            idx = idx.long()

        B, T = idx.shape
        device = idx.device

        # Bias accum in the module's compute dtype
        out_dtype = self.proj.weight.dtype
        bias = torch.zeros((B, T, self.vocab_size), device=device, dtype=out_dtype)

        # Aggregate memory embedding in dim-space, then project once.
        # We'll accumulate projected bias per-branch to avoid keeping huge (B,T,dim) for all n at once.
        for n_idx, n in enumerate(self.ns):
            if T < n:
                continue

            # windows: (B, L, n), where L = T - n + 1, aligned to positions t = n-1 .. T-1
            windows = idx.unfold(dimension=1, size=n, step=1).contiguous().to(torch.int64)
            L = windows.shape[1]

            # Sum embeddings across heads for this n
            mem_emb = torch.zeros((B, L, self.dim), device=device, dtype=out_dtype)

            for h in range(self.heads):
                hvals = _fnv1a_hash_ngram(windows, seed=int(self._seeds[h]), prime=int(self._primes[h]))
                buckets = torch.remainder(hvals, self.table_size).to(torch.int64)  # (B, L)
                mem_emb = mem_emb + self.tables[n_idx][h](buckets)  # gather -> (B, L, dim)

            mem_emb = mem_emb / float(self.heads)

            # Project to vocab bias and add into aligned positions
            branch_bias = self.proj(mem_emb)  # (B, L, vocab_size)
            bias[:, n - 1 :, :] = bias[:, n - 1 :, :] + branch_bias

        # Average across branches (optional, but keeps scale stable as you add ns)
        if len(self.ns) > 0:
            bias = bias / float(len(self.ns))

        return bias
