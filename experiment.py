"""Analytical KV-cache capacity for decoder-only transformer inference.

No model weights, temporary buffers, allocator overhead or GPU timing included.
"""
def kv_bytes(batch, tokens, layers, kv_heads, head_dim, bytes_per_element=2):
    values = (batch, tokens, layers, kv_heads, head_dim, bytes_per_element)
    if any(not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError("all dimensions must be positive integers")
    return 2 * batch * tokens * layers * kv_heads * head_dim * bytes_per_element


def main():
    layers, kv_heads, head_dim = 32, 8, 128
    print("Illustrative GQA configuration; cache capacity only")
    for tokens in (1024, 4096, 8192, 16384):
        size = kv_bytes(1, tokens, layers, kv_heads, head_dim)
        print(f"{tokens:>5} tokens: {size / 2**20:>7.1f} MiB")
    print("At fixed dimensions, doubling tokens doubles KV-cache bytes.")


if __name__ == "__main__":
    main()
