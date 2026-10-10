"""Pruebas de correctitud del Quadtree. Ejecutar: python -m unittest discover -s tests -v"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadtree import (build_quadtree, check_partition, decode, encode, encoded_size_bits,  # noqa: E402
                      iter_leaves, mse, reconstruct, split_region)
from quadtree.synthetic import blocks, detail, noise, uniform  # noqa: E402


def same_tree(a, b):
    if (a.x, a.y, a.w, a.h, a.color) != (b.x, b.y, b.w, b.h, b.color):
        return False
    if (a.children is None) != (b.children is None):
        return False
    if a.children is None:
        return True
    return len(a.children) == len(b.children) and all(same_tree(p, q) for p, q in zip(a.children, b.children))


class TestPartition(unittest.TestCase):
    def test_split_covers_region_exactly(self):
        for w in range(1, 20):
            for h in range(1, 20):
                cover = np.zeros((h, w), dtype=int)
                for x, y, ww, hh in split_region(0, 0, w, h):
                    self.assertGreater(ww, 0)
                    self.assertGreater(hh, 0)
                    if w * h > 1:
                        self.assertLess(ww * hh, w * h)  # progreso hacia el caso base
                    cover[y:y + hh, x:x + ww] += 1
                self.assertTrue((cover == 1).all(), (w, h))

    def test_leaves_tile_image_for_odd_sizes(self):
        for (h, w) in [(1, 1), (1, 17), (17, 1), (3, 5), (31, 33), (64, 64), (97, 61)]:
            img = detail(h, w)
            for tol in (0, 50, 1000):
                for mode in ("naive", "integral"):
                    self.assertTrue(check_partition(build_quadtree(img, tol, mode)), (h, w, tol, mode))


class TestCorrectness(unittest.TestCase):
    def test_exact_tolerance_is_lossless(self):
        for img in (blocks(50, 37), detail(33, 64), noise(20, 21), uniform(9, 9)):
            rec = reconstruct(build_quadtree(img, 0.0))
            self.assertTrue(np.array_equal(rec, img))

    def test_mse_bounded_by_tolerance(self):
        for img in (blocks(64, 64), detail(70, 45), noise(32, 32)):
            for tol in (1, 10, 100, 500, 3000):
                rec = reconstruct(build_quadtree(img, tol))
                err = float(((img.astype(float) - rec.astype(float)) ** 2).mean())
                self.assertLessEqual(err, tol + 1e-9, (tol, err))

    def test_naive_and_integral_build_identical_trees(self):
        for img in (blocks(61, 47), detail(40, 90), noise(16, 16)):
            for tol in (0, 20, 300):
                a = build_quadtree(img, tol, "naive")
                b = build_quadtree(img, tol, "integral")
                self.assertTrue(same_tree(a.root, b.root))
                self.assertEqual(a.stats.nodes, b.stats.nodes)

    def test_uniform_image_single_leaf(self):
        t = build_quadtree(uniform(37, 51), 0.0)
        self.assertEqual((t.stats.nodes, t.stats.leaves, t.stats.max_depth), (1, 1, 0))

    def test_node_counters_match_traversal(self):
        t = build_quadtree(detail(45, 77), 30.0)
        leaves = list(iter_leaves(t.root))
        self.assertEqual(len(leaves), t.stats.leaves)

        def count(n):
            return 1 + (0 if n.children is None else sum(count(c) for c in n.children))
        self.assertEqual(count(t.root), t.stats.nodes)

    def test_noise_exact_has_full_tree_on_power_of_two(self):
        t = build_quadtree(noise(16, 16), 0.0)
        self.assertEqual(t.stats.leaves, 256)
        self.assertEqual(t.stats.nodes, (4 * 256 - 1) // 3)
        self.assertEqual(t.stats.max_depth, 4)

    def test_naive_pixel_reads_formula_on_noise(self):
        # cada nivel (0..log2 L) recorre los N píxeles una vez
        t = build_quadtree(noise(32, 32), 0.0, "naive")
        self.assertEqual(t.stats.pixel_reads, 32 * 32 * (5 + 1))

    def test_grayscale_and_degenerate_shapes(self):
        gray = np.random.default_rng(1).integers(0, 256, size=(13, 9), dtype=np.uint8)
        t = build_quadtree(gray, 0.0)
        self.assertEqual(t.channels, 1)
        self.assertTrue(np.array_equal(reconstruct(t)[:, :, 0], gray))
        for shape in [(1, 1, 3), (1, 30, 3), (30, 1, 3)]:
            img = np.random.default_rng(2).integers(0, 256, size=shape, dtype=np.uint8)
            self.assertTrue(np.array_equal(reconstruct(build_quadtree(img, 0.0)), img))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            build_quadtree(uniform(4, 4), -1)
        with self.assertRaises(ValueError):
            build_quadtree(np.zeros((4, 4, 2), dtype=np.uint8), 0)


class TestCodec(unittest.TestCase):
    def test_roundtrip_and_size_formula(self):
        for img in (blocks(64, 64), detail(37, 53), noise(8, 8), uniform(5, 5)):
            for tol in (0, 40, 800):
                t = build_quadtree(img, tol)
                data = encode(t)
                self.assertEqual(len(data), (encoded_size_bits(t) + 7) // 8)
                t2 = decode(data)
                self.assertTrue(same_tree(t.root, t2.root))
                self.assertTrue(np.array_equal(reconstruct(t), reconstruct(t2)))

    def test_truncated_stream_raises(self):
        data = encode(build_quadtree(noise(8, 8), 0))
        with self.assertRaises(ValueError):
            decode(data[: len(data) // 2])


if __name__ == "__main__":
    unittest.main()

