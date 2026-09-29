"""Natural material continuation and immutable source-pixel safeguards."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'composite_natural_highlights.py'
SPEC = importlib.util.spec_from_file_location('natural_highlights', SCRIPT)
natural = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(natural)


class NaturalHighlightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source_path = self.directory / 'source.png'
        self.candidate_path = self.directory / 'candidate.png'
        self.mask_path = self.directory / 'mask.png'
        self.output_path = self.directory / 'output.png'
        yy, xx = np.indices((110, 180))
        texture = 2 * np.cos(2*np.pi*(xx/8 + yy/6)) + 2 * np.cos(2*np.pi*(xx/8 - yy/6))
        self.original = np.empty((110, 180, 4), dtype=np.uint8)
        self.original[:, :, :3] = np.rint(np.array([25, 30, 35]) + texture[:, :, None]).astype(np.uint8)
        self.original[:, :, 3] = 255
        self.mask = np.zeros((110, 180), dtype=np.uint8)
        self.mask[15:95, 55:90] = 255
        self.source = self.original.copy()
        self.source[self.mask > 0, :3] = 245
        self.candidate = self.original.copy()
        # AI low-frequency cloud and false structure must not become the base.
        cloud = np.maximum(0, 70 * np.cos(xx/20) + 80)
        self.candidate[:, :, :3] = np.clip(self.original[:, :, :3].astype(float) + cloud[:, :, None], 0, 255).astype(np.uint8)
        self.candidate[43:46, :, :3] = 0
        self.save()

    def save(self):
        Image.fromarray(self.source).save(self.source_path)
        Image.fromarray(self.candidate).save(self.candidate_path)
        Image.fromarray(self.mask).save(self.mask_path)

    def composite(self, **kwargs):
        options = dict(source=self.source_path, candidate=self.candidate_path, highlight_mask=self.mask_path,
                       output=self.output_path, texture_reference_boxes=[[115, 25, 165, 85]],
                       confirm_alignment=True, confirm_texture_references=True)
        options.update(kwargs)
        return natural.composite(**options)

    def test_grid_continues_with_source_color_without_ai_cloud_or_false_line(self):
        report = self.composite()
        output = np.array(Image.open(self.output_path).convert('RGBA'))
        core = output[23:87, 59:86, :3].astype(float)
        expected = self.original[23:87, 59:86, :3].astype(float)
        np.testing.assert_array_equal(output[self.mask == 0], self.source[self.mask == 0])
        self.assertLess(abs(core.mean()-expected.mean()), 1.)
        self.assertGreater(core.std(axis=(0, 1)).min(), .7)
        self.assertLess(core.std(axis=(0, 1)).max(), 3.)
        self.assertGreater(np.corrcoef(core[:, :, 0].ravel(), expected[:, :, 0].ravel())[0, 1], .75)
        self.assertLess(abs(core[20:23].mean()-core.mean()), 1.)
        self.assertEqual(report['outside_effective_mask_changed_pixels'], 0)
        self.assertEqual(report['texture_components'][0]['texture']['donor_pixels_copied'], 0)

    def test_alpha_edges_are_repaired_but_alpha_hidden_rgba_and_protection_stay_exact(self):
        self.source[20, 60] = [245, 245, 245, 128]
        self.candidate[20, 60] = [0, 0, 0, 0]  # Missing AA RGB may use source texture.
        self.source[22, 63] = [4, 78, 116, 0]
        protection = np.zeros_like(self.mask)
        protection[60:64, 65:69] = 255
        path = self.directory / 'protect.png'
        Image.fromarray(protection).save(path)
        self.save()
        before = self.source_path.read_bytes()
        report = self.composite(protect_mask=path)
        output = np.array(Image.open(self.output_path).convert('RGBA'))
        allowed = (self.mask > 0) & (self.source[:, :, 3] > 0) & (protection == 0)
        np.testing.assert_array_equal(output[~allowed], self.source[~allowed])
        np.testing.assert_array_equal(output[:, :, 3], self.source[:, :, 3])
        self.assertLess(output[20, 60, :3].max(), 80)
        self.assertEqual(self.source_path.read_bytes(), before)
        self.assertEqual(report['repaired_translucent_pixels'], 1)

    def test_edge_aware_base_removes_periodic_stripes_without_bright_rim_bleed(self):
        yy = np.arange(90)
        profile = np.repeat((30 + 2*np.cos(2*np.pi*yy/6))[:, None], 3, axis=1)
        profile[:2] = 220
        result = natural.profile_smooth(profile)
        self.assertLess(result[2:].max(), 33)
        self.assertLess(result[15:75].std(), .3)
        self.assertGreater(result[:2].min(), 215)

    def test_empty_mask_copies_original_file_bytes(self):
        self.mask[:] = 0
        self.save()
        report = self.composite(candidate=None, texture_reference_boxes=None, confirm_alignment=False,
                                confirm_texture_references=False)
        self.assertEqual(self.source_path.read_bytes(), self.output_path.read_bytes())
        self.assertTrue(report['byte_identical_copy'])

    def test_bad_reference_unconfirmed_candidate_and_opaque_gaps_fail_closed(self):
        cases = [dict(texture_reference_boxes=[[54, 14, 100, 98]]), dict(confirm_texture_references=False),
                 dict(confirm_alignment=False), dict(texture_reference_boxes=[[1, 1, 4, 4]])]
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.composite(**options)
                self.assertFalse(self.output_path.exists())
        self.candidate[40, 60, 3] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.composite()
        self.assertFalse(self.output_path.exists())

    def test_existing_destination_and_inputs_cannot_be_overwritten(self):
        self.output_path.write_bytes(b'keep')
        for path in (self.output_path, self.source_path, self.candidate_path, self.mask_path):
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                self.composite(output=path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
