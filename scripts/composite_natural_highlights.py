#!/usr/bin/env python3
"""Blend inspected AI highlight patches with reviewed source material texture.

Only registered highlight RGB with source alpha > 0 may change. All alpha,
hidden RGB, protected regions, and RGBA outside the mask remain exact. Texture
references must be reviewed clean interior material, without lines or defects.
The reconstructed material is an inference, not recovered hidden detail.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

_spec = importlib.util.spec_from_file_location('_highlight_compat', Path(__file__).with_name('composite_highlights.py'))
_compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compat)


def blur(array, sigma=1.3, axes=(0, 1)):
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    result = array.astype(float)
    for axis in axes:
        pad = [(0, 0)] * array.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(result, pad, mode='edge')
        result = np.zeros_like(result)
        for index, weight in enumerate(kernel):
            slices = [slice(None)] * array.ndim
            slices[axis] = slice(index, index + array.shape[axis])
            result += padded[tuple(slices)] * weight
    return result


def profile_smooth(profile):
    """Remove repeating fine texture without diffusing a real bright rim."""
    radius = 11
    padded = np.pad(profile, ((radius, radius), (0, 0)), mode='edge')
    numerator = np.zeros_like(profile, dtype=float)
    denominator = np.zeros((len(profile), 1))
    for offset in range(-radius, radius + 1):
        neighbor = padded[radius+offset:radius+offset+len(profile)]
        contrast = ((neighbor-profile).mean(axis=1) / 5.) ** 2
        weight = np.exp(-.5 * ((offset/3.5) ** 2 + contrast))[:, None]
        numerator += weight * neighbor
        denominator += weight
    return numerator / denominator


def components(mask):
    remaining = mask.copy()
    while remaining.any():
        start = tuple(np.argwhere(remaining)[0])
        stack = [start]
        remaining[start] = False
        points = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and remaining[yy, xx]:
                    remaining[yy, xx] = False
                    stack.append((yy, xx))
        yield np.asarray(points)


def _texture(src, candidate_detail, allowed, box, roi):
    x0, y0, x1, y1 = box
    rx0, ry0, rx1, ry1 = roi
    clean = src[ry0:ry1, rx0:rx1, :3]
    residual = clean - blur(clean)
    gray = residual.mean(axis=2)
    height, width = gray.shape
    yy, xx = np.indices((height, width))
    window = np.hanning(height)[:, None] * np.hanning(width)[None, :]
    power = np.abs(np.fft.fft2(gray * window)) ** 2
    frequency_y, frequency_x = np.fft.fftfreq(height), np.fft.fftfreq(width)
    frequencies = []
    # Two diagonal modes describe the inspected fine cross-hatched texture.
    # Only frequencies transfer from the donor; no donor pixel/edge is copied.
    for sign in (-1, 1):
        region = ((frequency_x[None, :] > .07) & (frequency_x[None, :] < .30)
                  & (frequency_y[:, None] * sign > .07) & (frequency_y[:, None] * sign < .30))
        py, px = np.unravel_index(np.argmax(power * region), power.shape)
        best = (0., float(frequency_x[px]), float(frequency_y[py]))
        for dx in np.linspace(-1 / width, 1 / width, 13):
            for dy in np.linspace(-1 / height, 1 / height, 13):
                fx, fy = float(frequency_x[px] + dx), float(frequency_y[py] + dy)
                score = abs(np.sum(gray * window * np.exp(-2j * np.pi * (fx * xx + fy * yy))))
                if score > best[0]:
                    best = (float(score), fx, fy)
        frequencies.append((best[1], best[2]))

    source_detail = src[:, :, :3] - blur(src[:, :, :3])
    fit_y = np.arange(y0 + 7, y1 - 7)
    fit_x = np.r_[np.arange(max(0, x0 - 16), max(0, x0 - 4)),
                     np.arange(min(src.shape[1], x1 + 4), min(src.shape[1], x1 + 16))]
    yy, xx = np.meshgrid(fit_y, fit_x, indexing='ij')
    yy, xx = yy.ravel(), xx.ravel()
    valid = ~allowed[yy, xx] & (src[yy, xx, 3] > 0)
    yy, xx = yy[valid], xx[valid]
    if len(yy) < 12:
        raise ValueError('Insufficient intact neighboring material to match texture phase')
    # Robust clipping stops adjacent material boundaries dominating the fit.
    target = np.clip(source_detail[yy, xx], -6, 6)
    # The nearby pattern may differ slightly from the distant clean reference.
    # Fit local frequency before phase, otherwise long strips cancel the grid.
    local_frequencies = []
    target_gray = target.mean(axis=1)
    for fx, fy in frequencies:
        best = (0., fx, fy)
        for dx in np.linspace(-.012, .012, 17):
            for dy in np.linspace(-.025, .025, 25):
                local_x, local_y = fx + dx, fy + dy
                score = abs(np.sum(target_gray * np.exp(-2j*np.pi*(local_x*xx + local_y*yy))))
                if score > best[0]:
                    best = (float(score), float(local_x), float(local_y))
        local_frequencies.append((best[1], best[2]))
    frequencies = local_frequencies
    design = np.stack([trig(2 * np.pi * (fx * xx + fy * yy))
                       for fx, fy in frequencies for trig in (np.cos, np.sin)], axis=1)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    out_y, out_x = np.mgrid[y0:y1, x0:x1]
    field = np.stack([trig(2 * np.pi * (fx * out_x + fy * out_y))
                      for fx, fy in frequencies for trig in (np.cos, np.sin)], axis=2) @ coefficients
    deviation = np.median(np.abs(residual - np.median(residual, axis=(0, 1))), axis=(0, 1)) * 1.4826
    texture_std = np.maximum(.4, deviation)
    fitted_rms = np.sqrt(np.mean(field ** 2, axis=(0, 1)))
    texture_gain = np.clip(texture_std / np.maximum(fitted_rms, .3), .5, 3.)
    field *= texture_gain
    detail = .9 * field + .1 * np.clip(candidate_detail, -2 * texture_std, 2 * texture_std)
    return detail, dict(reference_box=roi, frequencies_cycles_per_pixel=frequencies,
                        coefficients=coefficients.tolist(), source_texture_std=texture_std.tolist(),
                        texture_gain=texture_gain.tolist(), fitted_texture_rms=fitted_rms.tolist(),
                        ai_detail_fraction=.1, donor_pixels_copied=0, phase_sample_pixels=len(yy))


def blend_material(source, mapped, allowed, texture_reference_boxes):
    output = source.copy()
    parts = list(components(allowed))
    if len(texture_reference_boxes) not in (1, len(parts)):
        raise ValueError('Use one texture reference for all components or one per component in scan order')
    details = []
    for number, points in enumerate(parts):
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0) + 1
        transpose = (x1 - x0) > (y1 - y0)
        src = source.transpose(1, 0, 2) if transpose else source
        gen = mapped.transpose(1, 0, 2) if transpose else mapped
        mask = allowed.T if transpose else allowed
        roi = texture_reference_boxes[0 if len(texture_reference_boxes) == 1 else number]
        if transpose:
            y0, y1, x0, x1 = x0, x1, y0, y1
            roi = [roi[1], roi[0], roi[3], roi[2]]
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        if y1 - y0 <= 14 or x0 < 4 or x1 + 4 > src.shape[1]:
            raise ValueError('Repair needs a longer region with intact short-axis material on both sides')
        if mask[y0:y1, x0-4:x0].any() or mask[y0:y1, x1:x1+4].any():
            raise ValueError('Neighboring highlights overlap the material boundary samples')
        src, gen = src.astype(float), gen.astype(float)
        left = np.median(src[y0:y1, x0-4:x0, :3], axis=1)
        right = np.median(src[y0:y1, x1:x1+4, :3], axis=1)
        profile_left, profile_right = profile_smooth(left), profile_smooth(right)
        width = x1 - x0
        t = (np.arange(width) + 1) / (width + 1)
        base = profile_left[:, None, :] * (1-t[None, :, None]) + profile_right[:, None, :] * t[None, :, None]
        candidate_region = gen[y0:y1, x0:x1, :3]
        candidate_detail = candidate_region - blur(candidate_region, 1.0)
        # Robust nearby statistics normalize only the small AI detail term.
        # No pixel/edge from this nearby region is copied into the repair.
        if x1+width <= src.shape[1] and not mask[y0:y1, x1:x1+width].any():
            nearby = src[y0:y1, x1:x1+width, :3]
        elif x0-width >= 0 and not mask[y0:y1, x0-width:x0].any():
            nearby = src[y0:y1, x0-width:x0, :3]
        else:
            nearby = np.concatenate((src[y0:y1, max(0,x0-5):x0, :3],
                                     src[y0:y1, x1:min(src.shape[1],x1+5), :3]), axis=1)
        nearby_detail = nearby - blur(nearby, 1.0)
        inner = candidate_detail[5:-5, 3:-3] if y1-y0 > 10 and width > 6 else candidate_detail
        nearby_inner = nearby_detail[5:-5, 3:-3] if nearby_detail.shape[0] > 10 and nearby_detail.shape[1] > 6 else nearby_detail
        target_std = np.maximum(.25, np.median(np.abs(nearby_inner-np.median(nearby_inner,axis=(0,1))),axis=(0,1))*1.4826)
        candidate_std = np.maximum(.1, np.median(np.abs(inner-np.median(inner,axis=(0,1))),axis=(0,1))*1.4826)
        candidate_gain = np.clip(target_std/candidate_std, .2, 5)
        candidate_detail = np.clip(candidate_detail*candidate_gain, -4*target_std, 4*target_std)
        detail, texture_report = _texture(src, candidate_detail, mask, [x0, y0, x1, y1], roi)
        distance = np.minimum(np.arange(width)+1, width-np.arange(width))
        inward = np.clip(distance / 2, 0, 1)[None, :, None]
        side_texture = ((left-profile_left)[:, None, :] * (1-t[None, :, None])
                        + (right-profile_right)[:, None, :] * t[None, :, None])
        rgb = np.clip(base + inward * detail + (1-inward) * side_texture, 0, 255)
        target = output.transpose(1, 0, 2) if transpose else output
        select = mask[y0:y1, x0:x1]
        target[y0:y1, x0:x1, :3][select] = np.rint(rgb[select]).astype(np.uint8)
        details.append(dict(component=number, transposed=bool(transpose), working_box=[x0, y0, x1, y1],
                            pixels=int(len(points)), texture=texture_report))
    return output, details


def composite(source, candidate, highlight_mask, output, *, texture_reference_boxes=None,
              source_box=None, candidate_box=None, protect_mask=None,
              confirm_alignment=False, confirm_texture_references=False,
              candidate_alpha_threshold=192):
    source, output, highlight_mask = map(lambda path: Path(path).resolve(), (source, output, highlight_mask))
    candidate = Path(candidate).resolve() if candidate is not None else None
    protect_mask = Path(protect_mask).resolve() if protect_mask is not None else None
    inputs = {source, candidate, highlight_mask, protect_mask} - {None}
    if output in inputs or output.exists() or output.suffix.lower() != '.png':
        raise ValueError('Output must be a new PNG distinct from all inputs')
    if (isinstance(candidate_alpha_threshold, bool) or not isinstance(candidate_alpha_threshold, int)
            or not 1 <= candidate_alpha_threshold <= 255):
        raise ValueError('Candidate alpha threshold must be an integer from 1 to 255')
    with Image.open(source) as image:
        source_format = image.format
        source_image = image.convert('RGBA')
    src = np.array(source_image)
    highlighted = _compat._binary_mask(highlight_mask, source_image.size, 'Highlight')
    protected = (_compat._binary_mask(protect_mask, source_image.size, 'Protection') if protect_mask
                 else np.zeros(src.shape[:2], bool))
    allowed = highlighted & (src[:, :, 3] > 0) & ~protected
    output_pixels, details = src.copy(), []
    source_bounds = generated_bounds = candidate_size = None
    aspect_error = 0.
    if allowed.any():
        if candidate is None or not confirm_alignment:
            raise ValueError('Nonempty repair needs a candidate with visually confirmed alignment')
        if not texture_reference_boxes or not confirm_texture_references:
            raise ValueError('Review clean source texture references, then confirm them')
        references = [_compat._box(box, source_image.size, 'texture reference') for box in texture_reference_boxes]
        for x0, y0, x1, y1 in references:
            if min(x1-x0, y1-y0) < 16:
                raise ValueError('Texture references must be at least 16 pixels along each axis')
            if highlighted[y0:y1, x0:x1].any() or protected[y0:y1, x0:x1].any() or not np.all(src[y0:y1, x0:x1, 3] == 255):
                raise ValueError('Texture references must avoid highlights, protection, and translucent edges')
        with Image.open(candidate) as image:
            candidate_image = image.convert('RGBA')
        candidate_size = list(candidate_image.size)
        gen = np.array(candidate_image)
        if candidate_image.size == source_image.size and candidate_box is None and source_box is None:
            mapped = gen
            source_bounds = generated_bounds = (0, 0, source_image.width, source_image.height)
        else:
            source_bounds = _compat._box(source_box or source_image.getchannel('A').getbbox(), source_image.size, 'source')
            if candidate_box is None and np.all(gen[:, :, 3] >= _compat.ALPHA_THRESHOLD):
                raise ValueError('Opaque candidate needs a visually verified subject box')
            generated_bounds = _compat._box(candidate_box or _compat.largest_bbox(gen[:, :, 3] >= _compat.ALPHA_THRESHOLD), candidate_image.size, 'candidate')
            sx0, sy0, sx1, sy1 = source_bounds
            gx0, gy0, gx1, gy1 = generated_bounds
            aspect_error = abs(((gx1-gx0)/(gy1-gy0))/((sx1-sx0)/(sy1-sy0))-1)
            if aspect_error > _compat.MAX_ASPECT_ERROR:
                raise ValueError('Candidate aspect drift exceeds 6%; inspect or regenerate')
            mapped = np.zeros_like(src)
            mapped[sy0:sy1, sx0:sx1] = np.array(candidate_image.crop(generated_bounds).resize((sx1-sx0, sy1-sy0), Image.Resampling.LANCZOS))
        uncovered_core = allowed & (src[:, :, 3] == 255) & (mapped[:, :, 3] < candidate_alpha_threshold)
        if uncovered_core.any():
            raise ValueError('Candidate has unreliable coverage on opaque highlight pixels')
        output_pixels, details = blend_material(src, mapped, allowed, references)
    elif source_format != 'PNG':
        raise ValueError('Empty mask exact file copy requires PNG source')
    if not np.array_equal(output_pixels[~allowed], src[~allowed]) or not np.array_equal(output_pixels[:, :, 3], src[:, :, 3]):
        raise ValueError('Compositing violated original-pixel invariants')
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.natural-highlight-', suffix='.png', dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        if allowed.any():
            Image.fromarray(output_pixels).save(temporary)
        else:
            shutil.copy2(source, temporary)
        saved = np.array(Image.open(temporary).convert('RGBA'))
        if not np.array_equal(saved, output_pixels):
            raise ValueError('Saved PNG changed exact output pixels')
        report = dict(source=str(source), candidate=str(candidate) if candidate else None, output=str(output),
                      source_sha256=_compat.sha256(source), candidate_sha256=_compat.sha256(candidate) if candidate else None,
                      output_sha256=_compat.sha256(temporary), source_size=list(source_image.size), output_size=list(source_image.size),
                      candidate_size=candidate_size, source_box=source_bounds, candidate_box=generated_bounds, aspect_error=aspect_error,
                      highlight_mask=str(highlight_mask), highlight_mask_sha256=_compat.sha256(highlight_mask),
                      protected_mask=str(protect_mask) if protect_mask else None, highlighted_pixels=int(highlighted.sum()),
                      effective_highlight_pixels=int(allowed.sum()), changed_pixels=int(np.any(saved != src, axis=2).sum()),
                      outside_effective_mask_changed_pixels=0, protected_changed_pixels=0, alpha_exact=True, size_exact=True,
                      hidden_rgb_exact=True, byte_identical_copy=not bool(allowed.any()),
                      repaired_translucent_pixels=int((allowed & (src[:, :, 3] < 255)).sum()),
                      method='source-profile-with-clean-periodic-texture-and-limited-ai-detail',
                      candidate_alpha_threshold=candidate_alpha_threshold,
                      texture_reference_boxes=texture_reference_boxes, texture_components=details,
                      visual_fidelity='Inferred masked material; natural appearance and actual structure require visual review')
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--candidate')
    parser.add_argument('--highlight-mask', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--texture-references', help='JSON array of clean source [left,top,right,bottom] boxes')
    parser.add_argument('--source-box', nargs=4, type=int)
    parser.add_argument('--candidate-box', nargs=4, type=int)
    parser.add_argument('--protect-mask')
    parser.add_argument('--confirm-alignment', action='store_true')
    parser.add_argument('--confirm-texture-references', action='store_true')
    parser.add_argument('--candidate-alpha-threshold', type=int, default=192)
    parser.add_argument('--report', required=True)
    arguments = vars(parser.parse_args())
    report_path = Path(arguments.pop('report')).resolve()
    references_path = arguments.pop('texture_references')
    input_paths = {Path(arguments[key]).resolve() for key in ('source', 'candidate', 'highlight_mask', 'protect_mask', 'output') if arguments[key]}
    if references_path:
        input_paths.add(Path(references_path).resolve())
    if report_path.exists() or report_path in input_paths:
        parser.error('Report must be a new separate file')
    try:
        references = json.loads(Path(references_path).read_text()) if references_path else None
        report = composite(**arguments, texture_reference_boxes=references)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
