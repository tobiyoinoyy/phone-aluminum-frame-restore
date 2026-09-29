#!/usr/bin/env python3
"""Apply inspected AI highlights only; preserve every other source RGBA pixel.

Pillow and numpy are required. Masks are binary, source-sized images. Boxes use
(left, top, right, bottom), with right/bottom exclusive. Boundary blending uses
source surroundings and candidate gradients only within the permitted mask.
This script does not detect highlights or certify fidelity inside the mask.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

# Load the sibling implementation without relying on the caller's sys.path.
_spec = importlib.util.spec_from_file_location(
    "_highlight_geometry", Path(__file__).with_name("finalize_restore.py"))
_geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geometry)
largest_bbox = _geometry.largest_bbox
sha256 = _geometry.sha256

ALPHA_THRESHOLD = 192
MAX_ASPECT_ERROR = 0.06


def _binary_mask(path, size, label):
    with Image.open(path) as im:
        if im.size != size:
            raise ValueError(f"{label} mask must have exact source dimensions")
        rgba = np.array(im.convert("RGBA"))
    if (not np.all(rgba[:, :, 3] == 255)
            or not np.all(rgba[:, :, :3] == rgba[:, :, :1])
            or not np.all((rgba[:, :, 0] == 0) | (rgba[:, :, 0] == 255))):
        raise ValueError(f"{label} mask must be opaque binary black/white")
    return rgba[:, :, 0] == 255


def _box(value, size, label):
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"Invalid {label} box") from exc
    if (len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in values)
            or not _geometry.valid_box(values, size)):
        raise ValueError(f"Invalid {label} box")
    return tuple(int(v) for v in values)


def _inward_weights(permitted, feather):
    """Chebyshev-distance feather; padding is forbidden, never a bleed target."""
    if feather == 0:
        return permitted.astype(np.float32)
    weights = np.zeros(permitted.shape, dtype=np.float32)
    remaining = permitted.copy()
    for _ in range(feather + 1):
        weights += remaining.astype(np.float32) / (feather + 1)
        p = np.pad(remaining, 1, constant_values=False)
        remaining = np.logical_and.reduce([
            p[dy:dy + permitted.shape[0], dx:dx + permitted.shape[1]]
            for dy in range(3) for dx in range(3)])
    weights[~permitted] = 0
    return np.minimum(weights, 1)


def _boundary_blend(source_rgb, candidate_rgb, permitted, taper,
                    max_iterations, tolerance, anisotropy=16.0):
    """Poisson blend with source Dirichlet values and tapered AI gradients.

    Only unknown pixels inside ``permitted`` are solved. A candidate's broad
    color offset is discarded; its interior texture gradients remain. The
    source's overexposed interior is never used as a blending layer. Canvas
    edges have no fabricated neighbor, while excluded/protected pixels supply
    exact source boundary values. Narrow components emphasize their short-axis
    boundary so an existing bright end rim does not diffuse along the repair.
    Each component must have a fixed boundary.
    """
    yy, xx = np.nonzero(permitted)
    if not len(yy):
        return np.empty((0, 3)), dict(method="source-boundary-poisson", converged=True,
                                     components=0, iterations_rgb=[0, 0, 0],
                                     residual_rms_rgb=[0.0, 0.0, 0.0],
                                     tolerance_rms=tolerance, max_iterations=max_iterations,
                                     gradient_taper_pixels=taper, anisotropy=anisotropy,
                                     component_boundaries=[], clipped_values=0)
    # Crop computation, retaining one ring of source boundary pixels.
    y0, y1 = max(0, yy.min() - 1), min(permitted.shape[0], yy.max() + 2)
    x0, x1 = max(0, xx.min() - 1), min(permitted.shape[1], xx.max() + 2)
    mask = permitted[y0:y1, x0:x1]
    src = source_rgb[y0:y1, x0:x1].astype(np.float64)
    gen = candidate_rgb[y0:y1, x0:x1].astype(np.float64)
    py, px = np.nonzero(mask)
    n = len(py)
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[py, px] = np.arange(n)
    diagonal = np.zeros(n, dtype=np.float64)
    rhs = np.zeros((n, 3), dtype=np.float64)
    anchors = np.zeros(n, dtype=bool)
    neighbor_table = np.full((n, 4), -1, dtype=np.int64)
    directions = []
    weights = _inward_weights(mask, taper).astype(np.float64)
    for direction, (dy, dx) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        qy, qx = py + dy, px + dx
        valid = (qy >= 0) & (qy < mask.shape[0]) & (qx >= 0) & (qx < mask.shape[1])
        rows = np.flatnonzero(valid)
        qy, qx = qy[valid], qx[valid]
        neighbors = index[qy, qx]
        interior = neighbors >= 0
        inner_rows, inner_neighbors = rows[interior], neighbors[interior]
        neighbor_table[inner_rows, direction] = inner_neighbors
        directions.append((rows, neighbors, qy, qx, interior))
        outside_rows = rows[~interior]
        anchors[outside_rows] = True

    seen = np.zeros(n, dtype=bool)
    components = 0
    weight_x = np.ones(n, dtype=np.float64)
    weight_y = np.ones(n, dtype=np.float64)
    component_boundaries = []
    for start in range(n):
        if seen[start]:
            continue
        components += 1
        stack, anchored = [start], False
        points = []
        seen[start] = True
        while stack:
            point = stack.pop()
            points.append(point)
            anchored = anchored or bool(anchors[point])
            for neighbor in neighbor_table[point]:
                if neighbor >= 0 and not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        if not anchored:
            raise ValueError("Boundary blend has an unanchored component; no output written")
        points = np.asarray(points, dtype=np.int64)
        width = int(px[points].max() - px[points].min() + 1)
        height = int(py[points].max() - py[points].min() + 1)
        # On a narrow strip, propagate the undamaged same-row/column metal
        # across its short axis. This prevents a bright end rim from diffusing
        # far down the strip. Weights are constant per component, so the system
        # remains symmetric positive definite with its fixed source boundary.
        axis = "none"
        if height >= 1.25 * width:
            weight_x[points] = anisotropy
            axis = "x"
        elif width >= 1.25 * height:
            weight_y[points] = anisotropy
            axis = "y"
        component_boundaries.append(dict(pixels=len(points), width=width, height=height,
                                         emphasized_axis=axis))

    edges = []
    for direction, (rows, neighbors, qy, qx, interior) in enumerate(directions):
        strength_axis = (weight_y if direction < 2 else weight_x)[rows]
        diagonal[rows] += strength_axis
        edges.append((rows[interior], neighbors[interior], strength_axis[interior]))
        rhs[rows[~interior]] += strength_axis[~interior, None] * src[qy[~interior], qx[~interior]]
        # Zero guidance across the boundary; taper candidate gradients inward.
        # This avoids inheriting a candidate's artificial rectangular rim.
        strength = np.minimum(weights[py[rows], px[rows]], weights[qy, qx]) * strength_axis
        rhs[rows] += strength[:, None] * (gen[py[rows], px[rows]] - gen[qy, qx])

    def multiply(vector):
        result = diagonal * vector
        for rows, neighbors, strength_axis in edges:
            result[rows] -= strength_axis * vector[neighbors]
        return result

    result = np.empty((n, 3), dtype=np.float64)
    iterations, residuals = [], []
    for channel in range(3):
        x = gen[py, px, channel].copy()
        b = rhs[:, channel]
        residual = b - multiply(x)
        z = residual / diagonal
        direction = z.copy()
        rz = float(np.dot(residual, z))
        iteration = 0
        rms = float(np.sqrt(np.mean(residual * residual)))
        while rms > tolerance and iteration < max_iterations:
            ad = multiply(direction)
            denominator = float(np.dot(direction, ad))
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError("Boundary blend solver lost positive definiteness; no output written")
            step = rz / denominator
            x += step * direction
            residual -= step * ad
            iteration += 1
            rms = float(np.sqrt(np.mean(residual * residual)))
            if rms <= tolerance:
                break
            z = residual / diagonal
            next_rz = float(np.dot(residual, z))
            direction = z + (next_rz / rz) * direction
            rz = next_rz
        # Verify the true residual, not only recursively accumulated residuals.
        rms = float(np.sqrt(np.mean((b - multiply(x)) ** 2)))
        if not np.isfinite(x).all() or not math.isfinite(rms) or rms > tolerance:
            raise ValueError(f"Boundary blend did not converge for channel {channel} "
                             f"after {iteration} iterations (RMS {rms:.6g}); no output written")
        result[:, channel] = x
        iterations.append(iteration)
        residuals.append(rms)
    report = dict(method="source-boundary-poisson", converged=True, components=components,
                  iterations_rgb=iterations, residual_rms_rgb=residuals,
                  tolerance_rms=tolerance, max_iterations=max_iterations,
                  gradient_taper_pixels=taper,
                  anisotropy=anisotropy, component_boundaries=component_boundaries,
                  clipped_values=int(((result < 0) | (result > 255)).sum()))
    return np.clip(result, 0, 255), report


def composite(source, candidate, highlight_mask, output, *, candidate_box=None,
              source_box=None, protect_mask=None, feather=2, blend_mode="boundary",
              solver_max_iterations=2000, solver_tolerance=1e-5,
              boundary_anisotropy=16.0, confirm_alignment=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    mask_path = Path(highlight_mask).resolve()
    candidate_path = Path(candidate).resolve() if candidate is not None else None
    protect_path = Path(protect_mask).resolve() if protect_mask is not None else None
    inputs = {source, mask_path, candidate_path, protect_path} - {None}
    if output in inputs:
        raise ValueError("Output must not replace source, candidate, or masks")
    if output.exists():
        raise ValueError("Output already exists; use a new destination")
    if output.suffix.lower() != ".png":
        raise ValueError("Output must be PNG to preserve exact RGBA pixels")
    if isinstance(feather, bool) or not isinstance(feather, int) or not 0 <= feather <= 32:
        raise ValueError("Feather must be an integer from 0 to 32")
    if blend_mode not in ("boundary", "feather"):
        raise ValueError("Blend mode must be boundary or feather")
    if (isinstance(solver_max_iterations, bool) or not isinstance(solver_max_iterations, int)
            or solver_max_iterations < 1):
        raise ValueError("Solver maximum iterations must be a positive integer")
    if (isinstance(solver_tolerance, bool) or not isinstance(solver_tolerance, (int, float))
            or not math.isfinite(solver_tolerance) or not 0 < solver_tolerance <= 0.01):
        raise ValueError("Solver tolerance must be finite, positive, and at most 0.01")
    if (isinstance(boundary_anisotropy, bool) or not isinstance(boundary_anisotropy, (int, float))
            or not math.isfinite(boundary_anisotropy) or not 1 <= boundary_anisotropy <= 256):
        raise ValueError("Boundary anisotropy must be finite and from 1 to 256")
    with Image.open(source) as im:
        source_format = im.format
        src_im = im.convert("RGBA")
    src = np.array(src_im)
    highlights = _binary_mask(mask_path, src_im.size, "Highlight")
    protected = (_binary_mask(protect_path, src_im.size, "Protection") if protect_path
                 else np.zeros(src.shape[:2], dtype=bool))
    permitted = highlights & (src[:, :, 3] == 255) & ~protected
    out = src.copy()
    sb = gb = None
    candidate_size = None
    aspect_error = 0.0
    scales = [1.0, 1.0]
    coverage_count = 0
    blend_report = None
    generation_needed = bool(highlights.any())
    if generation_needed:
        if candidate_path is None:
            raise ValueError("Nonempty highlight mask requires an inspected candidate")
        if not confirm_alignment:
            raise ValueError("Inspect candidate alignment, then set confirm_alignment")
        with Image.open(candidate_path) as im:
            candidate_im = im.convert("RGBA")
        candidate_size = list(candidate_im.size)
        gen = np.array(candidate_im)
        if not np.any(src[:, :, 3]):
            raise ValueError("Source is fully transparent")
        if candidate_im.size == src_im.size and candidate_box is None and source_box is None:
            sb = gb = (0, 0, src_im.width, src_im.height)
            mapped = gen
        else:
            sb = _box(source_box if source_box is not None else src_im.getchannel("A").getbbox(), src_im.size, "source")
            if candidate_box is None and np.all(gen[:, :, 3] >= ALPHA_THRESHOLD):
                raise ValueError("Opaque candidate requires a visually verified candidate box")
            gb = _box(candidate_box if candidate_box is not None else largest_bbox(gen[:, :, 3] >= ALPHA_THRESHOLD), candidate_im.size, "candidate")
            sw, sh, gw, gh = sb[2] - sb[0], sb[3] - sb[1], gb[2] - gb[0], gb[3] - gb[1]
            aspect_error = abs((gw / gh) / (sw / sh) - 1)
            if aspect_error > MAX_ASPECT_ERROR:
                raise ValueError(f"Subject aspect drift {aspect_error:.3%} exceeds 6%; inspect or regenerate")
            mapped = np.zeros_like(src)
            mapped[sb[1]:sb[3], sb[0]:sb[2]] = np.array(candidate_im.crop(gb).resize((sw, sh), Image.Resampling.LANCZOS))
            scales = [sw / gw, sh / gh]
        unreliable = permitted & (mapped[:, :, 3] < ALPHA_THRESHOLD)
        coverage_count = int(unreliable.sum())
        if coverage_count:
            raise ValueError(f"Candidate has unreliable alpha coverage for {coverage_count} permitted highlight pixels; no output written")
        # Assign only permitted RGB. Alpha, hidden RGB, AA pixels, and every
        # excluded pixel remain from the original, with no global operations.
        if blend_mode == "boundary":
            blended, blend_report = _boundary_blend(src[:, :, :3], mapped[:, :, :3], permitted,
                                                    feather, solver_max_iterations, solver_tolerance,
                                                    boundary_anisotropy)
            out[permitted, :3] = np.rint(blended).astype(np.uint8)
        else:
            weights = _inward_weights(permitted, feather)
            w = weights[permitted, None]
            out[permitted, :3] = np.rint(src[permitted, :3].astype(np.float32) * (1 - w)
                                        + mapped[permitted, :3].astype(np.float32) * w).astype(np.uint8)
    elif source_box is not None or candidate_box is not None:
        raise ValueError("Empty highlight mask needs no geometry boxes")
    elif source_format != "PNG":
        raise ValueError("Byte-for-byte empty-mask copy requires a PNG source")

    # Validate all invariants before exposing the final path. An exclusive hard
    # link makes a concurrent destination creation fail rather than overwrite.
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".highlight-", suffix=".png", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        if not generation_needed:
            shutil.copy2(source, temporary)
        else:
            Image.fromarray(out).save(temporary, format="PNG")
        with Image.open(temporary) as im:
            saved = np.array(im.convert("RGBA"))
        alpha_exact = saved.shape == src.shape and np.array_equal(saved[:, :, 3], src[:, :, 3])
        outside_changed = int(np.any(saved[~permitted] != src[~permitted], axis=1).sum())
        protected_changed = int(np.any(saved[protected] != src[protected], axis=1).sum())
        if not alpha_exact or outside_changed or protected_changed:
            raise ValueError("Saved output violated exact-pixel invariants; no output written")
        report = {
            "source": str(source), "candidate": str(candidate_path) if candidate_path else None,
            "output": str(output), "source_sha256": sha256(source),
            "candidate_sha256": sha256(candidate_path) if generation_needed else None,
            "output_sha256": sha256(temporary), "source_size": list(src_im.size),
            "candidate_size": candidate_size, "output_size": [saved.shape[1], saved.shape[0]],
            "source_box": sb, "candidate_box": gb, "aspect_error": aspect_error,
            "max_aspect_error": MAX_ASPECT_ERROR, "scale_xy": scales,
            "generation_needed": generation_needed, "byte_identical_copy": not generation_needed,
            "alignment_visually_confirmed": bool(confirm_alignment) if generation_needed else None,
            "highlight_mask": str(mask_path), "highlight_mask_sha256": sha256(mask_path),
            "highlight_pixels": int(highlights.sum()), "effective_highlight_pixels": int(permitted.sum()),
            "protected_mask": str(protect_path) if protect_path else None,
            "protected_mask_sha256": sha256(protect_path) if protect_path else None,
            "protected_pixels": int(protected.sum()), "protected_changed_pixels": protected_changed,
            "outside_effective_mask_changed_pixels": outside_changed,
            "changed_pixels": int(np.any(saved != src, axis=2).sum()),
            "source_antialiased_edge_pixels_retained": int(((src[:, :, 3] > 0) & (src[:, :, 3] < 255)).sum()),
            "unreliably_covered_highlight_pixels": coverage_count,
            "blend_mode": blend_mode, "boundary_solver": blend_report,
            "feather_inward_pixels": feather if blend_mode == "feather" else 0,
            "alpha_exact": bool(alpha_exact), "size_exact": saved.shape == src.shape,
            "visual_fidelity": "Only masked RGB may change; highlight removal and details within masks require visual review",
        }
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--candidate")
    p.add_argument("--highlight-mask", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-box", nargs=4, type=int)
    p.add_argument("--candidate-box", nargs=4, type=int)
    p.add_argument("--protect-mask")
    p.add_argument("--feather", type=int, default=2)
    p.add_argument("--blend-mode", choices=("boundary", "feather"), default="boundary")
    p.add_argument("--solver-max-iterations", type=int, default=2000)
    p.add_argument("--solver-tolerance", type=float, default=1e-5)
    p.add_argument("--boundary-anisotropy", type=float, default=16.0)
    p.add_argument("--confirm-alignment", action="store_true")
    p.add_argument("--report", required=True)
    args = vars(p.parse_args())
    report_path = Path(args.pop("report")).resolve()
    inputs = {Path(args[x]).resolve() for x in ("source", "candidate", "highlight_mask", "protect_mask", "output") if args[x]}
    if report_path.exists() or report_path in inputs:
        p.error("Report must be a new separate file")
    try:
        report = composite(**args)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
