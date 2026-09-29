"""End-to-end checks using synthetic photographs, never real user photos."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'harmonize_aluminum.py'
SPEC = importlib.util.spec_from_file_location('harmonize_aluminum_under_test', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HarmonizeAluminumCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='aluminum-harmonize-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source.png'
        yy, xx = np.indices((176, 112))
        # Broad photographic exposure gradient plus visible alternating fine grain.
        light = .43 + .77 * yy / 175 + .03 * np.sin(xx / 19)
        grain = np.where((xx + yy) % 2, 2.4, -2.4)
        rgb = np.clip(np.rint(light[..., None] * [219, 119, 62]
                             + grain[..., None]), 0, 255).astype(np.uint8)
        alpha = np.full((176, 112), 255, dtype=np.uint8)
        alpha[:, :7] = 0
        alpha[:, -7:] = 0
        alpha[:, 7:12] = np.array([31, 75, 128, 180, 230])
        alpha[:, -12:-7] = np.array([230, 180, 128, 75, 31])
        alpha[:5] = 0
        alpha[-5:] = 0
        # A vivid label includes both light and dark detail, not merely a white box.
        label = ((xx[80:101, 45:67] + yy[80:101, 45:67]) % 3 == 0)
        rgb[80:101, 45:67] = np.where(label[..., None], [24, 202, 235], [217, 31, 184])
        self.original = np.dstack([rgb, alpha])
        Image.fromarray(self.original).save(self.source)
        self.original_hash = sha256(self.source)
        self.job = {
            'output_dir': str(self.root / 'photos'),
            'records_dir': str(self.root / 'records'),
            'defaults': {'sigma': [13, 9], 'strength': [.78, .72, .72],
                         'feather': 5, 'max_lab_shift': [48, 24, 24]},
            'views': [{
                'name': 'test-face', 'source': str(self.source),
                'reference_boxes': [[25, 105, 43, 120], [73, 105, 91, 120]],
                'protect': [{'type': 'rectangle', 'xy': [45, 80, 66, 100]}],
                'domains': [{'name': 'metal',
                             'shapes': [{'type': 'rectangle', 'xy': [8, 8, 103, 167]}]}],
            }],
        }
        self.job_path = self.root / 'job.json'

    def invoke(self):
        self.job_path.write_text(json.dumps(self.job))
        return subprocess.run([sys.executable, str(SCRIPT), '--job', str(self.job_path)],
                              capture_output=True, text=True, timeout=30)

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cli_preserves_source_geometry_protection_and_reduces_lighting(self):
        self.assert_success(self.invoke())
        destination = self.root / 'photos' / 'test-face.png'
        with Image.open(destination) as im:
            self.assertEqual(im.size, (112, 176))
            result = np.array(im.convert('RGBA'))
        self.assertEqual(sha256(self.source), self.original_hash)
        np.testing.assert_array_equal(result[..., 3], self.original[..., 3])
        self.assertTrue(np.any((result[..., 3] > 0) & (result[..., 3] < 255)))
        editable = np.zeros(self.original.shape[:2], dtype=bool)
        editable[8:168, 8:104] = True
        editable &= self.original[..., 3] > 0
        protected = np.zeros_like(editable)
        protected[80:101, 45:67] = True
        editable &= ~protected
        np.testing.assert_array_equal(result[~editable], self.original[~editable])
        np.testing.assert_array_equal(result[protected], self.original[protected])
        self.assertGreater(np.count_nonzero(np.any(result != self.original, axis=2)), 5000)

        before = MODULE.rgb_to_lab(self.original[..., :3])
        after = MODULE.rgb_to_lab(result[..., :3])
        def broad_gap(lab):
            return abs(float(np.median(lab[130:152, 25:43, 0])
                             - np.median(lab[24:46, 25:43, 0])))
        old_gap, new_gap = broad_gap(before), broad_gap(after)
        self.assertLess(new_gap, old_gap * .5)

        # A 2x2 alternating response removes a smooth illumination ramp and
        # checks that the fine grain survives instead of being blurred away.
        def detail(lab):
            luma = lab[38:138, 24:40, 0]
            return ((luma[:-1, :-1] - luma[1:, :-1]
                     - luma[:-1, 1:] + luma[1:, 1:]) / 4).ravel()
        old_detail, new_detail = detail(before), detail(after)
        retention = float(np.median(np.abs(new_detail)) / np.median(np.abs(old_detail)))
        correlation = float(np.corrcoef(old_detail, new_detail)[0, 1])
        self.assertGreater(retention, .65)
        self.assertLess(retention, 1.4)
        self.assertGreater(correlation, .98)
        print(json.dumps({'broad_lightness_gap_before': old_gap,
                          'broad_lightness_gap_after': new_gap,
                          'fine_detail_retention': retention,
                          'fine_detail_correlation': correlation}))
        self.assertEqual(sorted(p.name for p in (self.root / 'photos').iterdir()),
                         ['test-face.png'])
        report = json.loads((self.root / 'records' / 'validation.json').read_text())
        details = report['views']['test-face']
        for key in ('alpha_changed', 'protected_changed', 'outside_changed'):
            self.assertEqual(details[key], 0)
        self.assertEqual(report['uv_seam_validation'], 'not_performed_without_uv_correspondence')

    def test_external_masks_constrain_domain_and_protection_takes_priority(self):
        editable = np.zeros(self.original.shape[:2], dtype=np.uint8)
        # Nonzero grey levels are selections, including a separate value-1 island.
        # Extend past the geometric domain and over transparent pixels to verify
        # that an external mask can only narrow the source-coordinate domain.
        editable[20:155, :90] = 128
        editable[8:24, 95:110] = 1
        protected = np.zeros_like(editable)
        protected[35:53, 35:53] = 1
        external_editable = self.root / 'editable.png'
        external_protected = self.root / 'protected.png'
        Image.fromarray(editable).save(external_editable)
        Image.fromarray(protected).save(external_protected)
        self.job['views'][0].update(editable_mask=str(external_editable),
                                    protected_mask=str(external_protected))

        self.assert_success(self.invoke())
        with Image.open(self.root / 'photos' / 'test-face.png') as im:
            result = np.array(im.convert('RGBA'))
        expected_protection = protected > 0
        expected_protection[80:101, 45:67] = True  # Existing shape protection.
        expected_editable = np.zeros_like(expected_protection)
        expected_editable[8:168, 8:104] = True
        expected_editable &= ((editable > 0) & (self.original[..., 3] > 0)
                              & ~expected_protection)

        np.testing.assert_array_equal(result[~expected_editable],
                                      self.original[~expected_editable])
        np.testing.assert_array_equal(result[expected_protection],
                                      self.original[expected_protection])
        np.testing.assert_array_equal(result[..., 3], self.original[..., 3])
        self.assertGreater(np.count_nonzero(np.any(result != self.original, axis=2)), 3000)
        self.assertTrue(np.any(result[16, 98] != self.original[16, 98]))
        with Image.open(self.root / 'records' / 'test-face-editable.png') as im:
            np.testing.assert_array_equal(np.array(im) > 0, expected_editable)
        with Image.open(self.root / 'records' / 'test-face-protected.png') as im:
            np.testing.assert_array_equal(np.array(im) > 0, expected_protection)
        self.assertEqual(sha256(self.source), self.original_hash)

    def test_external_masks_with_wrong_dimensions_are_rejected(self):
        wrong_size = self.root / 'wrong-size.png'
        Image.new('L', (111, 176), 255).save(wrong_size)
        for field, error in [('protected_mask', 'Protection'),
                             ('editable_mask', 'Editable')]:
            with self.subTest(field=field):
                view = self.job['views'][0]
                view.pop('protected_mask', None)
                view.pop('editable_mask', None)
                view[field] = str(wrong_size)
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error + ' mask dimensions differ from source', result.stderr)
                self.assertFalse((self.root / 'photos' / 'test-face.png').exists())
                self.assertEqual(sha256(self.source), self.original_hash)

    def test_existing_delivery_is_refused_without_changes(self):
        self.assert_success(self.invoke())
        destination = self.root / 'photos' / 'test-face.png'
        before = sha256(destination)
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Output files already exist', result.stderr)
        self.assertEqual(sha256(destination), before)
        self.assertEqual(sha256(self.source), self.original_hash)

    def test_source_destination_is_refused_without_changes(self):
        self.job['output_dir'] = str(self.root)
        self.job['views'][0]['name'] = 'source'
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sha256(self.source), self.original_hash)

    def test_reports_inside_delivery_are_refused(self):
        self.job['records_dir'] = str(self.root / 'photos' / 'records')
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Records must live outside', result.stderr)
        self.assertFalse((self.root / 'photos').exists())


    def test_default_uses_per_face_lightness_and_keeps_source_chroma(self):
        second = self.root / 'second.png'
        a = np.zeros((80, 80, 4), dtype=np.uint8)
        a[:] = [205, 218, 225, 255]
        b = a.copy()
        b[..., :3] = [180, 172, 167]
        Image.fromarray(a).save(self.source)
        Image.fromarray(b).save(second)
        self.job['views'] = [
            {'name': name, 'source': str(path), 'reference_boxes': [[10, 10, 30, 30]],
             'domains': [{'shapes': [{'type': 'rectangle', 'xy': [0, 0, 79, 79]}]}]}
            for name, path in [('cool', self.source), ('warm', second)]]
        self.assert_success(self.invoke())
        for name, original in [('cool', a), ('warm', b)]:
            result = np.array(Image.open(self.root / 'photos' / (name+'.png')))
            # Pooling colors or L references would change these correctly lit faces.
            np.testing.assert_array_equal(result, original)

    def test_gamut_limit_reduces_l_shift_without_chroma_replacement(self):
        source = MODULE.rgb_to_lab(np.array([[230, 100, 35], [20, 55, 220]]))
        desired = source.copy()
        desired[:, 0] = [98, 95]
        limited, count = MODULE.preserve_chroma_in_gamut(desired, source)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(limited[:, 1:], source[:, 1:])
        self.assertTrue(np.all(limited[:, 0] > source[:, 0]))
        self.assertTrue(np.all(limited[:, 0] < desired[:, 0]))
        linear = MODULE.lab_to_linear_rgb(limited)
        self.assertGreaterEqual(linear.min(), -1.1e-7)
        self.assertLessEqual(linear.max(), 1+1.1e-7)
        saved = MODULE.rgb_to_lab(MODULE.lab_to_rgb(limited))
        self.assertLess(np.max(np.abs(saved[:, 1:]-source[:, 1:])), .8)

    def test_shared_color_requires_explicit_mode_and_remains_available(self):
        self.job['target_lab'] = [70, 0, 0]
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('require explicit color_mode shared_color', result.stderr)
        self.job['color_mode'] = 'shared_color'
        self.assert_success(self.invoke())
        report = json.loads((self.root / 'records' / 'validation.json').read_text())
        self.assertEqual(report['target']['mode'], 'shared_color')
        self.assertEqual(report['target']['target_lab'], [70, 0, 0])

    def test_complete_button_fit_exclusion_avoids_invented_halo(self):
        a = np.full((100, 100, 4), 255, dtype=np.uint8)
        a[..., :3] = [200, 210, 215]
        a[35:65, 40:60, :3] = [60, 65, 70]
        Image.fromarray(a).save(self.source)
        button = {'type': 'rectangle', 'xy': [35, 30, 64, 69]}
        item = {'source': str(self.source), 'exclude_from_fit': [button],
                'domains': [{'shapes': [{'type': 'rectangle', 'xy': [0, 0, 99, 99]}]}]}
        target = MODULE.rgb_to_lab(np.array([200, 210, 215]))
        result, allowed, _, _ = MODULE.process_view(item, target, {'sigma': [12, 12]})
        # Excluded pixels remain editable, but the dark button cannot lift adjacent metal.
        self.assertTrue(allowed[50, 50])
        np.testing.assert_array_equal(result, a)
        item.pop('exclude_from_fit')
        contaminated, _, _, _ = MODULE.process_view(item, target, {'sigma': [12, 12]})
        self.assertGreater(int(contaminated[50, 65, 0]), int(a[50, 65, 0]))

    def test_lightness_deadband_leaves_stable_exposure_unchanged(self):
        a = np.full((60, 60, 4), 255, dtype=np.uint8)
        a[..., :3] = [200, 210, 215]
        Image.fromarray(a).save(self.source)
        item = {'source': str(self.source), 'domains': [
            {'shapes': [{'type': 'rectangle', 'xy': [0, 0, 59, 59]}]}]}
        target = MODULE.rgb_to_lab(np.array([200, 210, 215]))
        target[0] += 1
        result, _, _, _ = MODULE.process_view(item, target, {'lightness_deadband': [1, 3]})
        np.testing.assert_array_equal(result, a)


if __name__ == '__main__':
    unittest.main()
