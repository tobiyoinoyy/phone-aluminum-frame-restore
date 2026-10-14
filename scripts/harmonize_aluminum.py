#!/usr/bin/env python3
"""Source-coordinate aluminum color harmonization; Pillow + NumPy only.

Inputs are a reviewed per-view job, material/protection masks, and optional AI
color reference. Never substitutes generated geometry or texture for the source.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb_to_lab(rgb):
    s = np.asarray(rgb, dtype=float) / 255
    lin = np.where(s <= .04045, s / 12.92, ((s + .055) / 1.055) ** 2.4)
    xyz = np.einsum('...j,ij->...i', lin, np.array([[.4124564, .3575761, .1804375],
                          [.2126729, .7151522, .0721750],
                          [.0193339, .1191920, .9503041]]))
    v = xyz / np.array([.95047, 1., 1.08883])
    f = np.where(v > (6/29)**3, np.cbrt(v), v / (3*(6/29)**2) + 4/29)
    return np.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]),
                     200*(f[...,1]-f[...,2])], axis=-1)


def lab_to_linear_rgb(lab):
    fy = (lab[...,0]+16)/116
    f = np.stack([fy+lab[...,1]/500, fy, fy-lab[...,2]/200], axis=-1)
    xyz = np.where(f > 6/29, f**3, 3*(6/29)**2*(f-4/29))
    xyz *= np.array([.95047, 1., 1.08883])
    lin = np.einsum('...j,ij->...i', xyz, np.linalg.inv(np.array([[.4124564,.3575761,.1804375],
                                      [.2126729,.7151522,.0721750],
                                      [.0193339,.1191920,.9503041]])))
    return lin


def lab_to_rgb(lab):
    lin = np.maximum(lab_to_linear_rgb(lab), 0)
    srgb = np.where(lin <= .0031308, lin*12.92, 1.055*lin**(1/2.4)-.055)
    return np.clip(np.rint(srgb*255), 0, 255).astype(np.uint8)


def preserve_chroma_in_gamut(desired, source):
    """Limit only L displacement; source a/b remain unchanged in float Lab."""
    result = np.array(desired, dtype=float, copy=True)
    result[..., 1:] = source[..., 1:]
    linear = lab_to_linear_rgb(result)
    bad = np.any((linear < -1e-7) | (linear > 1+1e-7), axis=-1)
    if bad.any():
        src, dst = source[bad], result[bad]
        delta = dst[..., 0] - src[..., 0]
        lo, hi = np.zeros(len(src)), np.ones(len(src))
        for _ in range(24):
            mid = (lo+hi)/2
            trial = dst.copy()
            trial[..., 0] = src[..., 0] + delta*mid
            rgb = lab_to_linear_rgb(trial)
            valid = np.all((rgb >= -1e-7) & (rgb <= 1+1e-7), axis=-1)
            lo = np.where(valid, mid, lo)
            hi = np.where(valid, hi, mid)
        dst[..., 0] = src[..., 0] + delta*lo
        result[bad] = dst
    return result, int(bad.sum())


def color_mode(config):
    mode = config.get('color_mode', 'source_lightness')
    if mode not in ('source_lightness', 'shared_color'):
        raise ValueError('Unknown color_mode: '+str(mode))
    return mode


def gaussian(arr, sigma):
    """Float separable Gaussian; reflection padding avoids dark image borders."""
    out = np.asarray(arr, dtype=float)
    sy, sx = sigma if isinstance(sigma, (list, tuple)) else (sigma, sigma)
    for axis, std in [(0, sy), (1, sx)]:
        if std <= 0:
            continue
        radius = max(1, int(np.ceil(3*std)))
        xx = np.arange(-radius, radius+1)
        kernel = np.exp(-.5*(xx/std)**2)
        kernel /= kernel.sum()
        pad = [(0, 0)]*out.ndim
        pad[axis] = (radius, radius)
        p = np.pad(out, pad, mode='reflect')
        out = np.apply_along_axis(lambda a: np.convolve(a, kernel, mode='valid'), axis, p)
    return out


def shapes_mask(size, shapes):
    im = Image.new('L', size)
    draw = ImageDraw.Draw(im)
    for shape in shapes:
        kind, coordinates = shape['type'], shape['xy']
        if kind == 'rectangle':
            draw.rectangle(coordinates, fill=255)
        elif kind == 'ellipse':
            draw.ellipse(coordinates, fill=255)
        elif kind == 'polygon':
            draw.polygon([tuple(p) for p in coordinates], fill=255)
        else:
            raise ValueError('Unknown shape: '+kind)
    return np.array(im) > 0


def supported_field(lab, fit, editable, sigma):
    """Never extrapolate an unsupported editable component from another surface.

    Shared by CLI and custom batch scripts. The fixed floor is a numerical
    guard, not a guarantee that the selected samples are the right material.
    Values outside reliable support are zero and must not enter smoothing.
    """
    support = gaussian(fit.astype(float), sigma)
    reliable = support >= .008
    missing = editable & ~reliable
    if missing.any():
        y, x = np.nonzero(missing)
        raise ValueError(f'Unsupported editable illumination: {len(x)} pixels; '
                         f'bounds [{x.min()}, {y.min()}, {x.max()}, {y.max()}]. '
                         'Split the component and fit its own reviewed source samples; '
                         'nearest-neighbor or median hole filling is not permitted.')
    low = np.stack([gaussian(lab[..., c]*fit, sigma)/np.maximum(support, 1e-9)
                    for c in range(3)], axis=-1)
    low[~reliable] = 0
    return low, support, reliable


def component_offset(source, accepted, component, reference, feather=2):
    """Repair one reviewed component from source texture with a constant Lab offset.

    The accepted image supplies color only in explicitly reviewed stable samples.
    It supplies no texture inside the repair. Outside the component is exact.
    """
    if source.shape != accepted.shape or source.ndim != 3 or source.shape[2] != 4:
        raise ValueError('Source and accepted image must have matching RGBA geometry')
    if not np.array_equal(source[..., 3], accepted[..., 3]):
        raise ValueError('Source and accepted alpha differ')
    component = np.asarray(component, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if component.shape != source.shape[:2] or reference.shape != component.shape:
        raise ValueError('Component masks must match source dimensions')
    if np.any(component & (source[..., 3] == 0)) or np.any(reference & ~component):
        raise ValueError('References must be inside the visible reviewed component')
    if reference.sum() < 20:
        raise ValueError('Insufficient reviewed component reference samples')
    lab = rgb_to_lab(source[..., :3])
    delta = np.median((rgb_to_lab(accepted[..., :3])-lab)[reference], axis=0)
    corrected = lab_to_rgb(lab + delta)
    weight = inward_weight(component, feather)
    dst = accepted.copy()
    mixed = np.clip(np.rint(corrected*weight[..., None]
                    + accepted[..., :3]*(1-weight[..., None])), 0, 255).astype(np.uint8)
    dst[component, :3] = mixed[component]
    return dst, {'method': 'source_texture_constant_component_lab_offset',
                 'offset_lab': delta.tolist(), 'reference_pixels': int(reference.sum()),
                 'component_pixels': int(component.sum()), 'feather': feather}


def inward_weight(mask, radius):
    """Discrete interior distance ramp, exactly zero outside reviewed mask."""
    if radius < 0:
        raise ValueError('Feather radius must be nonnegative')
    if radius == 0:
        return mask.astype(float)
    steps = max(1, int(radius))
    im = Image.fromarray(mask.astype(np.uint8)*255)
    total = mask.astype(float)
    for _ in range(steps):
        im = im.filter(ImageFilter.MinFilter(3))
        total += np.array(im)/255
    x = np.clip(total/(steps+1), 0, 1)
    return (x*x*(3-2*x))*mask


def target_color(job):
    colors = []
    for item in job['views']:
        a = np.array(Image.open(item['source']).convert('RGBA'))
        for box in item['reference_boxes']:
            x0,y0,x1,y1 = box
            roi = a[y0:y1,x0:x1]
            valid = roi[...,3] >= 250
            if not valid.any():
                raise ValueError('Empty reference box')
            colors.append(np.median(rgb_to_lab(roi[...,:3])[valid], axis=0))
    target = np.median(colors, axis=0)
    info = {'source_reference_lab': target.tolist()}
    ref = job.get('ai_reference')
    if ref:
        a = np.array(Image.open(ref['path']).convert('RGBA'))
        x0,y0,x1,y1 = ref['box']
        roi = a[y0:y1,x0:x1]
        valid = roi[...,3] >= 250
        if not valid.any():
            raise ValueError('Empty AI reference box')
        candidate = np.median(rgb_to_lab(roi[...,:3])[valid], axis=0)
        limit = np.asarray(ref.get('max_lab_shift', [8., 5., 5.]))
        delta = np.clip(candidate-target, -limit, limit)
        target = target + float(ref.get('weight', .25))*delta
        info.update(ai_reference_lab=candidate.tolist(), ai_reference_sha256=digest(ref['path']))
    if 'target_lab' in job:
        target = np.array(job['target_lab'], dtype=float)
    info['target_lab'] = target.tolist()
    if not np.isfinite(target).all():
        raise ValueError('Non-finite target color')
    info['target_rgb'] = lab_to_rgb(target).tolist()
    return target, info


def supported_contour_field(lab, fit, editable, alpha, sigma):
    """Tangent/normal source sampling; include curved rims and alpha fringes.

    Requires SciPy (included by phone-texture-preprocessor). No nearest-neighbor
    fill or lowering support threshold when a material is not represented.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
    tangent, normal = np.asarray(sigma,dtype=float)
    if not np.isfinite([tangent,normal]).all() or tangent<=0 or normal<=0:
        raise ValueError('Contour sigma must be two finite positive values')
    inside=alpha>=128
    dist=distance_transform_edt(np.pad(inside,1))[1:-1,1:-1]-distance_transform_edt(~inside)
    gy,gx=np.gradient(gaussian_filter(dist,3));norm=np.maximum(np.hypot(gx,gy),1e-8)
    nx,ny=gx/norm,gy/norm;y,x=np.nonzero(editable)
    num=np.zeros((len(x),3));den=np.zeros(len(x));total=0.
    for t in np.linspace(-2.5*tangent,2.5*tangent,19):
        for n in np.linspace(-2*normal,2*normal,5):
            inset=np.maximum(0,5-dist[y,x])
            coordinates=[y+nx[y,x]*t+ny[y,x]*(n+inset),x-ny[y,x]*t+nx[y,x]*(n+inset)]
            weight=np.exp(-.5*(t/tangent)**2-.5*(n/normal)**2);total+=weight
            den+=weight*map_coordinates(fit.astype(float),coordinates,order=1,mode='constant',cval=0)
            for channel in range(3):
                num[:,channel]+=weight*map_coordinates(lab[...,channel]*fit,coordinates,order=1,mode='constant',cval=0)
    strength=den/total
    if (strength<.008).any():raise ValueError('Unsupported editable contour illumination')
    if not np.isfinite(num).all():raise ValueError('Non-finite contour illumination')
    low=np.zeros_like(lab);support=np.zeros(alpha.shape);low[y,x]=num/den[:,None];support[y,x]=strength
    return low,support,support>=.008


def process_view(item, target, defaults):
    path = Path(item['source'])
    before_hash = digest(path)
    src = np.array(Image.open(path).convert('RGBA'))
    h,w = src.shape[:2]
    protect = shapes_mask((w,h), item.get('protect', []))
    if item.get('protected_mask'):
        external = np.array(Image.open(item['protected_mask']).convert('L'))
        if external.shape != (h,w):
            raise ValueError('Protection mask dimensions differ from source')
        protect |= external > 0
    if item.get('protection_policy', 'reviewed_regions') not in ('sticker_only','reviewed_regions'):
        raise ValueError('Unknown protection policy')
    if item.get('protection_policy') == 'sticker_only':
        if item.get('source_sha256') != before_hash or not item.get('sticker_mask'):
            raise ValueError('Sticker-only policy requires source SHA256 and sticker mask')
        paper = np.array(Image.open(item['sticker_mask']).convert('L')) > 0
        if paper.shape != (h,w):
            raise ValueError('Sticker mask dimensions differ')
        visible = src[...,3] > 0
        if np.any((protect ^ paper) & visible):
            raise ValueError('Sticker-only policy rejects protection outside actual paper or unprotected paper')
    reviewed = np.ones((h,w), dtype=bool)
    if item.get('editable_mask'):
        external = np.array(Image.open(item['editable_mask']).convert('L'))
        if external.shape != (h,w):
            raise ValueError('Editable mask dimensions differ from source')
        reviewed = external > 0
    required = np.zeros((h,w), dtype=bool)
    if item.get('required_editable_mask'):
        external = np.array(Image.open(item['required_editable_mask']).convert('L'))
        if external.shape != (h,w):
            raise ValueError('Required editable mask dimensions differ from source')
        required = (external > 0) & (src[...,3] > 0)
        if np.any(required & protect):
            raise ValueError('Required editable pixels remain protected; revise obsolete protection')
    exclude = shapes_mask((w,h), item.get('exclude_from_fit', [])) | protect
    lab = rgb_to_lab(src[...,:3])
    mode = color_mode(defaults)
    out_lab = lab.copy()
    allowed_union = np.zeros((h,w), dtype=bool)
    domains = []
    for domain in item['domains']:
        material = shapes_mask((w,h), domain['shapes']) & (src[...,3] > 0) & reviewed
        allowed = material & ~protect
        if np.any(allowed_union & allowed):
            raise ValueError('Material domains must not overlap')
        allowed_union |= allowed
        fit = material & ~exclude & (src[...,3] >= 240)
        if fit.sum() < 20:
            raise ValueError('Insufficient reviewed material samples')
        sigma = domain.get('sigma', defaults.get('sigma', [18,18]))
        field = domain.get('field','gaussian')
        if field == 'contour':
            low, support, reliable = supported_contour_field(lab, fit, allowed, src[...,3], sigma)
        elif field == 'gaussian':
            low, support, reliable = supported_field(lab, fit, allowed, sigma)
        else:
            raise ValueError('Unknown illumination field')
        strengths = np.array(domain.get('strength', defaults.get('strength', [.78,.72,.72])))
        if mode == 'source_lightness':
            strengths = strengths.copy()
            strengths[1:] = 0
        shift = (target-low)*strengths
        limits = np.array(defaults.get('max_lab_shift', [48,24,24]))
        shift = np.clip(shift, -limits, limits)
        # Smooth all correction components, including across excluded features.
        # Protected pixels themselves are reinstated exactly after conversion.
        smooth = domain.get('correction_smooth', 3)
        shift = (gaussian(shift*reliable[..., None], smooth)
                 / np.maximum(gaussian(reliable.astype(float), smooth)[..., None], 1e-9))
        weight = inward_weight(allowed, domain.get('feather', defaults.get('feather', 5)))
        out_lab += shift*weight[...,None]
        domains.append({'name':domain.get('name','metal'), 'editable_pixels':int(allowed.sum()),
                        'fit_pixels':int(fit.sum()),'field':field,'strength':strengths.tolist(),
                        'minimum_editable_support':float(support[allowed].min()) if allowed.any() else None,
                        'unsupported_editable_pixels':0})
    if np.any(required & ~allowed_union):
        raise ValueError('Required editable pixels missing from material domains or editable mask')
    gamut_limited = 0
    if mode == 'source_lightness':
        threshold = np.asarray(defaults.get('lightness_deadband', [0, 0]), dtype=float)
        if (threshold.shape != (2,) or not np.isfinite(threshold).all()
                or threshold[0] < 0 or threshold[1] < threshold[0]
                or (threshold[1] > 0 and threshold[1] == threshold[0])):
            raise ValueError('lightness_deadband needs 0 <= start < end, or [0, 0]')
        if threshold[1] > 0:
            delta = out_lab[..., 0]-lab[..., 0]
            t = np.clip((np.abs(delta)-threshold[0])/(threshold[1]-threshold[0]), 0, 1)
            out_lab[..., 0] = lab[..., 0]+delta*t*t*(3-2*t)
        out_lab, gamut_limited = preserve_chroma_in_gamut(out_lab, lab)
    dst = src.copy()
    converted = lab_to_rgb(out_lab)
    dst[allowed_union,:3] = converted[allowed_union]
    assert np.array_equal(dst[...,3],src[...,3])
    assert np.array_equal(dst[protect],src[protect])
    assert np.array_equal(dst[~allowed_union],src[~allowed_union])
    if digest(path) != before_hash:
        raise RuntimeError('Source changed during processing')
    return dst, allowed_union, protect, {'source':str(path),'source_sha256':before_hash,
        'size':[w,h], 'domains':domains, 'color_mode':mode,
        'gamut_limited_pixels':gamut_limited,
        'required_editable_pixels':int(required.sum()),
        'required_editable_missing':0,
        'changed_pixels':int(np.any(dst != src,axis=-1).sum()),
        'alpha_changed':0,'protected_changed':0,'outside_changed':0}


def run(job_path):
    job = json.loads(Path(job_path).read_text())
    output = Path(job['output_dir'])
    records = Path(job['records_dir'])
    paths = [Path(v['source']).resolve() for v in job['views']]
    outputs = [output/(v['name']+'.png') for v in job['views']]
    if len(set(outputs)) != len(outputs):
        raise ValueError('Duplicate face names')
    if any(p.exists() for p in outputs):
        raise FileExistsError('Output files already exist; choose a new version directory')
    if any(p.resolve() in paths for p in outputs):
        raise ValueError('Source overwrite is prohibited')
    if records.resolve() == output.resolve() or output.resolve() in records.resolve().parents:
        raise ValueError('Records must live outside the photo delivery directory')
    output.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)
    mode = color_mode(job)
    defaults = dict(job.get('defaults', {}), color_mode=mode)
    if mode == 'source_lightness':
        if 'target_lab' in job or 'ai_reference' in job:
            raise ValueError('Shared color targets require explicit color_mode shared_color')
        targets = [target_color({'views': [item]}) for item in job['views']]
        info = {'mode': mode, 'per_view_source_references': {
            item['name']: entry[1] for item, entry in zip(job['views'], targets)}}
    else:
        target, info = target_color(job)
        targets = [(target, info)]*len(job['views'])
        info['mode'] = mode
    report = {'target':info, 'uv_seam_validation':'not_performed_without_uv_correspondence', 'views':{}}
    for item, dest, (target, _) in zip(job['views'], outputs, targets):
        dst, allowed, protect, details = process_view(item,target,defaults)
        src_im = Image.open(item['source'])
        save_args = {'icc_profile':src_im.info['icc_profile']} if 'icc_profile' in src_im.info else {}
        Image.fromarray(dst).save(dest, **save_args)
        saved = np.array(Image.open(dest).convert('RGBA'))
        src = np.array(Image.open(item['source']).convert('RGBA'))
        if saved.shape != src.shape or not np.array_equal(saved[...,3],src[...,3]):
            raise AssertionError('Saved output geometry/alpha differs')
        if not np.array_equal(saved[~allowed],src[~allowed]) or not np.array_equal(saved[protect],src[protect]):
            raise AssertionError('Saved output protection differs')
        if digest(item['source']) != details['source_sha256']:
            raise AssertionError('Source hash changed')
        details.update(output=str(dest),output_sha256=digest(dest))
        report['views'][item['name']] = details
        Image.fromarray(allowed.astype(np.uint8)*255).save(records/(item['name']+'-editable.png'))
        Image.fromarray(protect.astype(np.uint8)*255).save(records/(item['name']+'-protected.png'))
    (records/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True,help='Reviewed JSON job; see material-harmonization reference')
    run(parser.parse_args().job)
