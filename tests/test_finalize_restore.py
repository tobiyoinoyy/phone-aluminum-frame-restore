"""Synthetic regression tests; no real phone photos or external services needed.

Run from the skill directory:
    python3 -m unittest discover -s tests -v

Dependencies are the same as finalize_restore.py: Pillow and numpy.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "finalize_restore.py"
SPEC = importlib.util.spec_from_file_location("finalize_restore", SCRIPT)
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


class FinalizeRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source_path = self.directory / "source.png"
        self.generated_path = self.directory / "generated.png"
        self.output_path = self.directory / "final.png"

        # A tightly cropped phone with original texture, transparent corners,
        # and exactly 228 partially transparent antialiased edge pixels.
        rng = np.random.default_rng(998)
        self.source = rng.integers(0, 256, (81, 41, 4), dtype=np.uint8)
        self.source[:, :, 3] = 255
        for rows in (slice(0, 2), slice(-2, None)):
            for columns in (slice(0, 2), slice(-2, None)):
                self.source[rows, columns, 3] = 0
        self.source[0, 2:-2, 3] = 128
        self.source[-1, 2:-2, 3] = 128
        self.source[2:-2, 0, 3] = 128
        self.source[2:-2, -1, 3] = 128
        Image.fromarray(self.source).save(self.source_path)

        # A 2x subject on a much larger transparent canvas plus isolated debris.
        generated = np.zeros((238, 148, 4), dtype=np.uint8)
        generated[47:209, 31:113] = [70, 90, 110, 255]
        generated[5:7, 5:7] = [255, 0, 0, 255]
        Image.fromarray(generated).save(self.generated_path)

    def finalize(self, **kwargs):
        options = dict(source=self.source_path, generated=self.generated_path,
                       output=self.output_path, confirm_alignment=True)
        options.update(kwargs)
        return restore.finalize(**options)

    def read_output(self):
        with Image.open(self.output_path) as result:
            return np.array(result.convert("RGBA"))

    def test_original_canvas_alpha_and_file_are_preserved(self):
        before = self.source_path.read_bytes()
        report = self.finalize()
        output = self.read_output()
        self.assertEqual(output.shape, self.source.shape)
        self.assertEqual(report["output_size"], [41, 81])
        np.testing.assert_array_equal(output[:, :, 3], self.source[:, :, 3])
        np.testing.assert_array_equal(output[self.source[:, :, 3] == 0],
                                      self.source[self.source[:, :, 3] == 0])
        # This also proves the result is not an unchanged copy of the source.
        np.testing.assert_array_equal(output[20, 20], [70, 90, 110, 255])
        self.assertEqual(self.source_path.read_bytes(), before)
        self.assertTrue(report["alpha_exact"])
        self.assertTrue(report["size_exact"])

    def test_sticker_original_rgba_is_exact_even_with_external_feather(self):
        mask = np.zeros(self.source.shape[:2], dtype=np.uint8)
        mask[52:64, 12:27] = 255
        mask_path = self.directory / "sticker-mask.png"
        Image.fromarray(mask).save(mask_path)
        report = self.finalize(protect_mask=mask_path, feather=3)
        output = self.read_output()
        np.testing.assert_array_equal(output[mask == 255],
                                      self.source[mask == 255])
        self.assertEqual(report["protected_pixels"], 180)
        self.assertEqual(report["protected_changed_pixels"], 0)
        np.testing.assert_array_equal(output[20, 20], [70, 90, 110, 255])

    def test_crop_ignores_large_transparent_margins_and_isolated_noise(self):
        report = self.finalize()
        self.assertEqual(tuple(report["source_box"]), (0, 0, 41, 81))
        self.assertEqual(tuple(report["generated_box"]), (31, 47, 113, 209))
        self.assertEqual(report["scale_xy"], [0.5, 0.5])
        self.assertEqual(report["aspect_error"], 0)
        self.assertEqual(report["source_opaque_gap_pixels_retained"], 0)

    def test_aspect_drift_and_overwrites_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "aspect drift"):
            self.finalize(generated_box=(31, 47, 110, 160))
        self.assertFalse(self.output_path.exists())
        existing_path = self.directory / "existing.png"
        existing_path.write_bytes(b"existing output must remain untouched")
        for path in (self.source_path, self.generated_path, existing_path):
            with self.subTest(destination=path.name):
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "replace|already exists"):
                    self.finalize(output=path)
                self.assertEqual(path.read_bytes(), before)

    def test_invalid_protection_masks_are_rejected_before_output(self):
        cases = (
            ("wrong-size", np.zeros((10, 10), dtype=np.uint8), "dimensions"),
            ("nonbinary", np.full((81, 41), 128, dtype=np.uint8), "binary"),
            ("empty", np.zeros((81, 41), dtype=np.uint8), "empty"),
        )
        for name, pixels, message in cases:
            with self.subTest(mask=name):
                mask_path = self.directory / f"{name}.png"
                Image.fromarray(pixels).save(mask_path)
                with self.assertRaisesRegex(ValueError, message):
                    self.finalize(protect_mask=mask_path)
                self.assertFalse(self.output_path.exists())

    def test_antialiased_edges_and_opaque_gaps_have_separate_exact_counts(self):
        # An equal-size generation avoids interpolation: a 5x7 transparent gap
        # must retain exactly 35 opaque source pixels, independently of AA edges.
        generated = np.full((81, 41, 4), [70, 90, 110, 255], dtype=np.uint8)
        generated[20:25, 18:25, 3] = 0
        Image.fromarray(generated).save(self.generated_path)
        report = self.finalize()
        output = self.read_output()
        edge = (self.source[:, :, 3] > 0) & (self.source[:, :, 3] < 255)
        np.testing.assert_array_equal(output[edge], self.source[edge])
        np.testing.assert_array_equal(output[20:25, 18:25],
                                      self.source[20:25, 18:25])
        self.assertEqual(report["source_antialiased_edge_pixels_retained"], 228)
        self.assertEqual(report["source_opaque_gap_pixels_retained"], 35)


if __name__ == "__main__":
    unittest.main()
