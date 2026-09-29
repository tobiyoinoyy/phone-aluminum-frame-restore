"""Paired-UV seam tests use temporary synthetic images, never user photos."""
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


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
SCRIPT = SCRIPTS / 'match_uv_seams.py'
SPEC = importlib.util.spec_from_file_location('seam_test_color', SCRIPTS / 'harmonize_aluminum.py')
COLOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLOR)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MatchUvSeamsCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='aluminum-seam-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.originals, self.hashes, self.protected, self.editable = {}, {}, {}, {}
        self.job = {'scene': str(self.root / 'scene.json'),
                    'seam_chains': str(self.root / 'chains.json'),
                    'output_dir': str(self.root / 'photos'),
                    'records_dir': str(self.root / 'records'),
                    'width_units': .25, 'guard_pixels': 6, 'sigma': 3,
                    'protection_feather': 3, 'views': {}}
        self.job_path = self.root / 'job.json'
        # Explicit reviewed chains need no topology inference from this scene.
        (self.root / 'scene.json').write_text('{}')
        (self.root / 'chains.json').write_text(json.dumps({'chains': [{
            'names': ['darker', 'lighter'], 'edges': [101, 102],
            'arc': [0, .5, 1], 'sides': [
                {'name': 'darker', 'points': [[34, 24], [34, 96], [34, 168]],
                 'normals': [[1, 0]] * 3, 'px_per_unit': 160},
                {'name': 'lighter', 'points': [[158, 24], [158, 96], [158, 168]],
                 'normals': [[-1, 0]] * 3, 'px_per_unit': 160},
            ],
        }]}))
        yy, xx = np.indices((192, 192))
        grain = np.where((xx + yy) % 2, 3., -3.)
        for name, base, seam_x in [('darker', [178, 89, 46], 34),
                                   ('lighter', [218, 134, 75], 158)]:
            # Both surfaces carry the same fine pattern and broad variation;
            # a different base color creates a strong artificial UV seam.
            ramp = yy / 191 * 8 + np.sin(xx / 35) * 2
            rgb = np.clip(np.rint(np.array(base) + ramp[..., None]
                                 + grain[..., None]), 0, 255).astype(np.uint8)
            alpha = np.full((192, 192), 255, dtype=np.uint8)
            alpha[:, :8] = 0
            alpha[:, -8:] = 0
            alpha[:8] = 0
            alpha[-8:] = 0
            alpha[:, 8:12] = [31, 90, 160, 230]
            alpha[:, -12:-8] = [230, 160, 90, 31]
            # Also put partial-alpha pixels inside the corrected band, so the
            # alpha check cannot pass merely because its border is untouched.
            alpha[122:132, seam_x - 2:seam_x + 3] = 128
            protected = (yy >= 80) & (yy < 101) & (abs(xx - seam_x) < 11)
            label = np.where(((xx + yy) % 3 == 0)[..., None],
                             [25, 220, 232], [224, 24, 175])
            rgb[protected] = label[protected]
            rgba = np.dstack([rgb, alpha])
            editable = (xx >= 16) & (xx < 176) & (yy >= 16) & (yy < 176)
            source = self.root / (name + '.png')
            allow_path = self.root / (name + '-editable.png')
            protect_path = self.root / (name + '-protected.png')
            Image.fromarray(rgba).save(source)
            Image.fromarray(editable.astype(np.uint8) * 255).save(allow_path)
            Image.fromarray(protected.astype(np.uint8) * 255).save(protect_path)
            self.originals[name] = rgba
            self.hashes[name] = sha256(source)
            self.protected[name] = protected
            self.editable[name] = editable
            self.job['views'][name] = {'source': str(source),
                                      'editable_mask': str(allow_path),
                                      'protected_mask': str(protect_path)}

    def invoke(self):
        self.job_path.write_text(json.dumps(self.job))
        return subprocess.run([sys.executable, str(SCRIPT), '--job', str(self.job_path)],
                              capture_output=True, text=True, timeout=30)

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_sources_unchanged(self):
        for name, spec in self.job['views'].items():
            self.assertEqual(sha256(spec['source']), self.hashes[name])

    def test_preserves_pixels_and_detail_while_reducing_paired_seam(self):
        self.assert_success(self.invoke())
        report = json.loads((self.root / 'records' / 'validation.json').read_text())
        outputs = {}
        yy, xx = np.indices((192, 192))
        for name, source in self.originals.items():
            with Image.open(self.root / 'photos' / (name + '.png')) as im:
                self.assertEqual(im.size, (192, 192))
                self.assertEqual(im.mode, 'RGBA')
                result = np.array(im)
            outputs[name] = result
            np.testing.assert_array_equal(result[..., 3], source[..., 3])
            np.testing.assert_array_equal(result[self.protected[name]], source[self.protected[name]])
            np.testing.assert_array_equal(result[~self.editable[name]], source[~self.editable[name]])
            np.testing.assert_array_equal(result[source[..., 3] == 0], source[source[..., 3] == 0])
            # Independently bound the permissible seam band in pixel space,
            # rather than trusting only the mask generated by the script.
            xs = (xx >= 28) & (xx <= 74) if name == 'darker' else (xx >= 118) & (xx <= 164)
            bounding_band = xs & (yy >= 18) & (yy <= 174)
            np.testing.assert_array_equal(result[~bounding_band], source[~bounding_band])
            effective = np.array(Image.open(self.root / 'records' / (name + '-seam-mask.png'))) > 0
            self.assertFalse(np.any(effective & ~bounding_band))
            np.testing.assert_array_equal(result[~effective], source[~effective])
            changed = np.any(result != source, axis=-1)
            self.assertGreater(int(changed.sum()), 1000)
            self.assertTrue(np.any(changed & (source[..., 3] == 128)))
            self.assertEqual(report['checks'][name]['source_sha256'], self.hashes[name])
            for key in ('alpha_changed', 'outside_changed', 'protected_changed'):
                self.assertEqual(report['checks'][name][key], 0)

            # The mixed finite difference removes a smooth color correction;
            # high correlation checks the original fine pattern survives.
            x0 = 44 if name == 'darker' else 130
            def detail(rgba):
                luma = COLOR.rgb_to_lab(rgba[36:72, x0:x0 + 16, :3])[..., 0]
                return ((luma[:-1, :-1] - luma[1:, :-1]
                         - luma[:-1, 1:] + luma[1:, 1:]) / 4).ravel()
            old_detail, new_detail = detail(source), detail(result)
            retention = float(np.median(abs(new_detail)) / np.median(abs(old_detail)))
            self.assertGreater(retention, .7)
            self.assertLess(retention, 1.3)
            self.assertGreater(float(np.corrcoef(old_detail, new_detail)[0, 1]), .98)

        self.assert_sources_unchanged()
        seam = report['seams'][0]
        self.assertGreater(seam['before_mean_deltaE76'], 10)
        self.assertLess(seam['after_mean_deltaE76'], seam['before_mean_deltaE76'] * .25)
        self.assertGreater(seam['metric_samples'], 100)
        # Cross-check the improvement directly on corresponding stored pixels,
        # outside the protected label and partial-alpha portion of the seam.
        rows = np.r_[36:70, 108:120, 140:160]
        def direct_delta(images):
            a = COLOR.rgb_to_lab(images['darker'][rows, 34, :3])
            b = COLOR.rgb_to_lab(images['lighter'][rows, 158, :3])
            return float(np.linalg.norm(a - b, axis=-1).mean())
        self.assertLess(direct_delta(outputs), direct_delta(self.originals) * .25)
        self.assertEqual(sorted(p.name for p in (self.root / 'photos').iterdir()),
                         ['darker.png', 'lighter.png'])

    def test_existing_output_is_rejected_without_changes(self):
        self.assert_success(self.invoke())
        previous = {p.name: sha256(p) for p in (self.root / 'photos').iterdir()}
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Choose new output directory', result.stderr)
        self.assertEqual({p.name: sha256(p) for p in (self.root / 'photos').iterdir()}, previous)
        self.assert_sources_unchanged()

    def test_source_destination_is_rejected_without_changes(self):
        self.job['output_dir'] = str(self.root)
        self.job['records_dir'] = str(self.root.parent / (self.root.name + '-unused-records'))
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assert_sources_unchanged()
        self.assertFalse(Path(self.job['records_dir']).exists())

    def test_records_inside_delivery_are_rejected(self):
        self.job['records_dir'] = str(self.root / 'photos' / 'records')
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Keep records outside delivery', result.stderr)
        self.assertFalse((self.root / 'photos').exists())
        self.assert_sources_unchanged()


if __name__ == '__main__':
    unittest.main()
