"""Grouped-query attention KV-cache capacity model (CPU-only).

Assumptions: decoder-only transformer; all layers same dimensions; cache
stores K and V, with no compression or metadata overhead.
Run: python gqa_tradeoff.py
"""
def kv_cache_bytes(batch, tokens, layers, kv_heads, head_dim, dtype_bytes):
    if min(batch, tokens, layers, kv_heads, head_dim, dtype_bytes) <= 0:
        raise ValueError("All parameters must be positive")
    return 2 * batch * tokens * layers * kv_heads * head_dim * dtype_bytes


def main():
    batch, tokens, layers, head_dim, dtype_bytes = 1, 4096, 32, 128, 2
    baseline = kv_cache_bytes(batch, tokens, layers, 32, head_dim, dtype_bytes)
    print("KV heads | cache MiB | reduction vs 32 KV heads")
    for kv_heads in (32, 16, 8, 4, 1):
        size = kv_cache_bytes(batch, tokens, layers, kv_heads, head_dim, dtype_bytes)
        print(f"{kv_heads:>8} | {size / 2**20:>9.1f} | {baseline / size:>7.1f}x")
    assert baseline == 512 * 2**20
    assert kv_cache_bytes(1, 4096, 32, 8, 128, 2) == 128 * 2**20


if __name__ == "__main__":
    main()
