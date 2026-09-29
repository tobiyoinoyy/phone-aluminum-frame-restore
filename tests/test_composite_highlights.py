"""Exact-pixel and fail-closed tests for source-preserving highlight compositing."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "composite_highlights.py"
SPEC = importlib.util.spec_from_file_location("composite_highlights", SCRIPT)
highlights = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(highlights)


class HighlightCompositeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source_path = self.directory / "source.png"
        self.candidate_path = self.directory / "candidate.png"
        self.mask_path = self.directory / "mask.png"
        self.output_path = self.directory / "output.png"
        rng = np.random.default_rng(32)
        self.source = rng.integers(0, 256, (60, 30, 4), dtype=np.uint8)
        self.source[:, :, 3] = 255
        self.source[0, :, 3] = 128
        self.source[-1, :3, 3] = 0
        Image.fromarray(self.source).save(self.source_path)
        self.candidate = np.full(self.source.shape, [20, 40, 60, 255], dtype=np.uint8)
        Image.fromarray(self.candidate).save(self.candidate_path)
        self.mask = np.zeros(self.source.shape[:2], dtype=np.uint8)
        self.mask[0:20, 0:20] = 255
        Image.fromarray(self.mask).save(self.mask_path)

    def composite(self, **kwargs):
        options = dict(source=self.source_path, candidate=self.candidate_path,
                       highlight_mask=self.mask_path, output=self.output_path,
                       confirm_alignment=True, feather=0, blend_mode="feather")
        options.update(kwargs)
        return highlights.composite(**options)

    def output(self):
        return np.array(Image.open(self.output_path).convert("RGBA"))

    def test_only_highlight_rgb_changes_original_alpha_and_rest_exact(self):
        before = self.source_path.read_bytes()
        report = self.composite()
        output = self.output()
        editable = (self.mask == 255) & (self.source[:, :, 3] == 255)
        np.testing.assert_array_equal(output[~editable], self.source[~editable])
        np.testing.assert_array_equal(output[:, :, 3], self.source[:, :, 3])
        np.testing.assert_array_equal(output[editable], self.candidate[editable])
        self.assertEqual(output.shape, self.source.shape)
        self.assertEqual(self.source_path.read_bytes(), before)
        self.assertEqual(report["outside_effective_mask_changed_pixels"], 0)
        self.assertTrue(report["alpha_exact"] and report["size_exact"])

    def test_inward_feather_does_not_bleed_even_at_canvas_edge(self):
        report = self.composite(feather=3)
        output = self.output()
        editable = (self.mask == 255) & (self.source[:, :, 3] == 255)
        np.testing.assert_array_equal(output[~editable], self.source[~editable])
        np.testing.assert_array_equal(output[8, 8], self.candidate[8, 8])
        expected = np.rint(self.source[8, 0, :3] * .75 + self.candidate[8, 0, :3] * .25).astype(np.uint8)
        np.testing.assert_array_equal(output[8, 0, :3], expected)
        self.assertEqual(report["outside_effective_mask_changed_pixels"], 0)

    def test_protected_sticker_overlap_is_exact_with_inward_feather(self):
        protected = np.zeros(self.mask.shape, dtype=np.uint8)
        protected[6:11, 6:11] = 255
        protect_path = self.directory / "protect.png"
        Image.fromarray(protected).save(protect_path)
        report = self.composite(protect_mask=protect_path, feather=2)
        output = self.output()
        np.testing.assert_array_equal(output[protected == 255], self.source[protected == 255])
        self.assertEqual(report["protected_changed_pixels"], 0)
        self.assertEqual(report["effective_highlight_pixels"], 355)
        self.assertFalse(np.array_equal(output[5, 5, :3], self.candidate[5, 5, :3]))

    def test_padded_double_size_candidate_uses_subject_box_not_canvas(self):
        generated = np.zeros((160, 100, 4), dtype=np.uint8)
        generated[20:140, 20:80] = [20, 40, 60, 255]
        generated[1:3, 1:3] = [255, 0, 0, 255]
        Image.fromarray(generated).save(self.candidate_path)
        report = self.composite()
        self.assertEqual(report["candidate_box"], (20, 20, 80, 140))
        self.assertEqual(report["scale_xy"], [.5, .5])
        np.testing.assert_array_equal(self.output()[10, 10], [20, 40, 60, 255])

    def test_empty_mask_needs_no_candidate_and_copies_exact_file_bytes(self):
        Image.fromarray(np.zeros(self.mask.shape, dtype=np.uint8)).save(self.mask_path)
        before = self.source_path.read_bytes()
        report = self.composite(candidate=None, confirm_alignment=False)
        self.assertEqual(self.output_path.read_bytes(), before)
        self.assertEqual(report["source_sha256"], report["output_sha256"])
        self.assertFalse(report["generation_needed"])
        self.assertTrue(report["byte_identical_copy"])
        self.assertEqual(report["changed_pixels"], 0)

    def synthetic(self, expected, candidate, mask):
        """Replace the selected area by glare while retaining known surroundings."""
        self.mask = mask.astype(np.uint8) * 255
        self.source = expected.copy()
        self.source[mask, :3] = 250
        Image.fromarray(self.source).save(self.source_path)
        Image.fromarray(candidate).save(self.candidate_path)
        Image.fromarray(self.mask).save(self.mask_path)

    def test_boundary_is_default_and_constant_surroundings_have_no_patch_jump(self):
        expected = np.full(self.source.shape, [38, 62, 97, 255], dtype=np.uint8)
        mask = np.zeros(self.source.shape[:2], dtype=bool)
        mask[10:50, 5:25] = True
        candidate = np.full(self.source.shape, [160, 175, 200, 255], dtype=np.uint8)
        self.synthetic(expected, candidate, mask)
        report = highlights.composite(self.source_path, self.candidate_path, self.mask_path,
                                      self.output_path, feather=6, confirm_alignment=True)
        np.testing.assert_array_equal(self.output(), expected)
        self.assertEqual(report["blend_mode"], "boundary")
        solver = report["boundary_solver"]
        self.assertTrue(solver["converged"])
        self.assertEqual(solver["components"], 1)
        self.assertEqual(solver["gradient_taper_pixels"], 6)
        self.assertTrue(all(value <= solver["tolerance_rms"] for value in solver["residual_rms_rgb"]))
        self.assertTrue(all(value > 0 for value in solver["iterations_rgb"]))

    def test_boundary_recovers_linear_color_gradient_from_surroundings(self):
        yy, xx = np.indices(self.source.shape[:2])
        expected = np.stack((30 + xx + 2 * yy, 40 + 2 * xx + yy,
                             80 + xx, np.full_like(xx, 255)), axis=2).astype(np.uint8)
        mask = np.zeros(self.source.shape[:2], dtype=bool)
        mask[8:53, 5:24] = True
        candidate = np.full(self.source.shape, [90, 90, 90, 255], dtype=np.uint8)
        self.synthetic(expected, candidate, mask)
        self.composite(blend_mode="boundary", feather=6)
        np.testing.assert_array_equal(self.output(), expected)

    def test_boundary_retains_candidate_interior_texture_without_its_color_offset(self):
        expected = np.full(self.source.shape, [50, 70, 90, 255], dtype=np.uint8)
        candidate = np.full(self.source.shape, [110, 140, 190, 255], dtype=np.uint8)
        texture = np.array([[0, 6, -3], [4, 12, 5], [-2, 3, 0]], dtype=np.int16)
        for channel in range(3):
            expected[25:28, 14:17, channel] = expected[25:28, 14:17, channel].astype(np.int16) + texture
            candidate[25:28, 14:17, channel] = candidate[25:28, 14:17, channel].astype(np.int16) + texture
        mask = np.zeros(self.source.shape[:2], dtype=bool)
        mask[5:55, 3:27] = True
        self.synthetic(expected, candidate, mask)
        self.composite(blend_mode="boundary", feather=4)
        np.testing.assert_array_equal(self.output(), expected)

    def test_boundary_bright_end_rim_does_not_spread_down_narrow_strip(self):
        source = np.full((60, 30, 3), 30, dtype=np.uint8)
        mask = np.zeros((60, 30), dtype=bool)
        mask[10:50, 11:19] = True
        source[9, 11:19] = 230  # Real one-pixel rim outside the editable strip.
        source[mask] = 250
        candidate = np.full_like(source, 100)
        anisotropic, report = highlights._boundary_blend(source, candidate, mask, 6, 2000, 1e-5)
        isotropic, _ = highlights._boundary_blend(source, candidate, mask, 6, 2000, 1e-5, 1)
        result = source.astype(np.float64)
        result[mask] = anisotropic
        legacy = source.astype(np.float64)
        legacy[mask] = isotropic
        self.assertLess(result[13, 14, 0], 32)
        self.assertGreater(legacy[13, 14, 0], 80)
        self.assertEqual(report["component_boundaries"][0]["emphasized_axis"], "x")
        # Rotation chooses the corresponding short axis, with identical result.
        rotated, rotated_report = highlights._boundary_blend(source.transpose(1, 0, 2),
            candidate.transpose(1, 0, 2), mask.T, 6, 2000, 1e-5)
        rotated_result = source.transpose(1, 0, 2).astype(np.float64)
        rotated_result[mask.T] = rotated
        np.testing.assert_allclose(rotated_result.transpose(1, 0, 2), result, atol=1e-5)
        self.assertEqual(rotated_report["component_boundaries"][0]["emphasized_axis"], "y")

    def test_boundary_disconnected_masks_preserve_protected_and_antialiased_pixels(self):
        self.mask[40:48, 4:12] = 255
        Image.fromarray(self.mask).save(self.mask_path)
        protected = np.zeros(self.mask.shape, dtype=np.uint8)
        protected[6:11, 6:11] = 255
        path = self.directory / "protect.png"
        Image.fromarray(protected).save(path)
        report = self.composite(blend_mode="boundary", protect_mask=path, feather=6)
        output = self.output()
        editable = (self.mask == 255) & (self.source[:, :, 3] == 255) & (protected == 0)
        np.testing.assert_array_equal(output[~editable], self.source[~editable])
        np.testing.assert_array_equal(output[:, :, 3], self.source[:, :, 3])
        self.assertEqual(report["protected_changed_pixels"], 0)
        self.assertEqual(report["boundary_solver"]["components"], 2)

    def test_boundary_wide_87_by_47_repair_uses_vertical_neighbors(self):
        source = np.full((80, 120, 3), 30, dtype=np.uint8)
        mask = np.zeros((80, 120), dtype=bool)
        mask[11:58, 10:97] = True  # Representative side-face repair dimensions.
        source[11:58, 9] = 230
        source[mask] = 250
        candidate = np.full_like(source, 100)
        pixels, report = highlights._boundary_blend(source, candidate, mask, 6, 2000, 1e-5, 64)
        isotropic, _ = highlights._boundary_blend(source, candidate, mask, 6, 2000, 1e-5, 1)
        result = source.astype(np.float64)
        result[mask] = pixels
        legacy = source.astype(np.float64)
        legacy[mask] = isotropic
        self.assertEqual(report["component_boundaries"][0]["emphasized_axis"], "y")
        self.assertLess(result[34, 20, 0], 32)
        self.assertGreater(legacy[34, 20, 0], 140)
        self.assertLess(result[mask, 0].max(), 230)  # No new brighter hotspot.
        np.testing.assert_array_equal(result[~mask], source[~mask])

    def test_boundary_fails_closed_for_unanchored_mask(self):
        self.source[:, :, 3] = 255
        Image.fromarray(self.source).save(self.source_path)
        Image.fromarray(np.full(self.mask.shape, 255, dtype=np.uint8)).save(self.mask_path)
        with self.assertRaisesRegex(ValueError, "unanchored"):
            self.composite(blend_mode="boundary")
        self.assertFalse(self.output_path.exists())

    def test_boundary_fails_closed_when_solver_does_not_converge(self):
        with self.assertRaisesRegex(ValueError, "did not converge"):
            self.composite(blend_mode="boundary", solver_max_iterations=1)
        self.assertFalse(self.output_path.exists())

    def test_unreliable_candidate_coverage_fails_before_output(self):
        self.candidate[8:10, 8:10, 3] = 0
        Image.fromarray(self.candidate).save(self.candidate_path)
        with self.assertRaisesRegex(ValueError, "unreliable alpha coverage for 4"):
            self.composite()
        self.assertFalse(self.output_path.exists())

    def test_mask_validation_rejects_wrong_size_gray_and_transparent(self):
        for array in (np.zeros((2, 3), dtype=np.uint8), np.full(self.mask.shape, 128, dtype=np.uint8),
                      np.zeros((*self.mask.shape, 4), dtype=np.uint8)):
            for argument in ("highlight_mask", "protect_mask"):
                path = self.directory / "bad-mask.png"
                Image.fromarray(array).save(path)
                with self.subTest(shape=array.shape, argument=argument):
                    with self.assertRaisesRegex(ValueError, "dimensions|binary"):
                        self.composite(**{argument: path})
                    self.assertFalse(self.output_path.exists())

    def test_geometry_inspection_and_output_overwrite_guards(self):
        cases = [dict(confirm_alignment=False), dict(candidate=None), dict(feather=-1),
                 dict(blend_mode="unsupported"), dict(solver_max_iterations=0),
                 dict(solver_tolerance=float("nan")), dict(solver_tolerance=1),
                 dict(boundary_anisotropy=0), dict(boundary_anisotropy=float("nan")),
                 dict(candidate_box=(0, 0, 10, 60)), dict(source_box=(-1, 0, 30, 60)),
                 dict(candidate_box=(0.5, 0, 30, 60)), dict(candidate_box=(0, 0, 30)),
                 dict(output=self.directory / "lossy.jpg")]
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.composite(**options)
                self.assertFalse(self.output_path.exists())
        self.output_path.write_bytes(b"do not overwrite")
        for output in (self.source_path, self.candidate_path, self.mask_path, self.output_path):
            before = output.read_bytes()
            with self.assertRaises(ValueError):
                self.composite(output=output)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
