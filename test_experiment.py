import unittest
from experiment import kv_bytes


class KVCacheTests(unittest.TestCase):
    def test_one_token(self):
        self.assertEqual(kv_bytes(1, 1, 1, 1, 4, 2), 16)

    def test_context_doubles(self):
        self.assertEqual(kv_bytes(1, 20, 4, 2, 8), 2 * kv_bytes(1, 10, 4, 2, 8))

    def test_gqa_reduces_capacity(self):
        self.assertEqual(kv_bytes(1, 10, 4, 2, 8) * 4, kv_bytes(1, 10, 4, 8, 8))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            kv_bytes(0, 10, 4, 2, 8)


if __name__ == "__main__":
    unittest.main()
