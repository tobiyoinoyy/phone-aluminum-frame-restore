#!/usr/bin/env python3
"""Restore source canvas/alpha and copy protected source pixels after generation.

Requires Pillow and numpy. Does not relight, redraw, denoise, or sharpen.
All boxes are pixel (left, top, right, bottom), with right/bottom exclusive.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def largest_bbox(mask):
    """8-connected component bbox using row runs; ignores isolated alpha debris."""
    parents, sizes, boxes = [], [], []

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    previous = []
    for y, row in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
        current = []
        j = 0
        for x0, x1 in edges.reshape(-1, 2):
            idx = len(parents)
            parents.append(idx)
            sizes.append(int(x1 - x0))
            boxes.append([int(x0), y, int(x1), y + 1])
            while j < len(previous) and previous[j][1] < x0:
                j += 1
            k = j
            while k < len(previous) and previous[k][0] <= x1:
                a, b = root(idx), root(previous[k][2])
                if a != b:
                    parents[b] = a
                    sizes[a] += sizes[b]
                    boxes[a] = [min(boxes[a][0], boxes[b][0]),
                                min(boxes[a][1], boxes[b][1]),
                                max(boxes[a][2], boxes[b][2]),
                                max(boxes[a][3], boxes[b][3])]
                k += 1
            current.append((x0, x1, idx))
        previous = current
    roots = [i for i in range(len(parents)) if parents[i] == i]
    if not roots:
        raise ValueError("No opaque subject found")
    return tuple(boxes[max(roots, key=lambda i: sizes[i])])


def valid_box(box, size):
    x0, y0, x1, y1 = box
    return 0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]


def finalize(source, generated, output, *, source_box=None, generated_box=None,
             protect_mask=None, feather=0, alpha_threshold=192,
             max_aspect_error=0.06, confirm_alignment=False):
    paths = [Path(x).resolve() for x in (source, generated, output)]
    if paths[2] in paths[:2]:
        raise ValueError("Output must not replace source or raw generated image")
    if paths[2].exists():
        raise ValueError("Output already exists; use a new destination")
    if not confirm_alignment:
        raise ValueError("Inspect silhouette and structural anchors, then confirm alignment")
    src_im, gen_im = [Image.open(x).convert("RGBA") for x in paths[:2]]
    src, gen = np.array(src_im), np.array(gen_im)
    if not np.any(src[:, :, 3]):
        raise ValueError("Source is fully transparent")
    if generated_box is None and np.all(gen[:, :, 3] >= alpha_threshold):
        raise ValueError("Generated image is opaque: specify a visually verified subject box")
    sb = tuple(source_box or src_im.getchannel("A").getbbox())
    gb = tuple(generated_box or largest_bbox(gen[:, :, 3] >= alpha_threshold))
    if not valid_box(sb, src_im.size) or not valid_box(gb, gen_im.size):
        raise ValueError("Invalid subject box")
    sw, sh = sb[2] - sb[0], sb[3] - sb[1]
    gw, gh = gb[2] - gb[0], gb[3] - gb[1]
    aspect_error = abs((gw / gh) / (sw / sh) - 1)
    if aspect_error > max_aspect_error:
        raise ValueError(f"Subject aspect drift {aspect_error:.3%} exceeds limit; inspect or regenerate")
    # Resize subject only. Pillow RGBA resize uses premultiplied alpha.
    mapped = np.array(gen_im.crop(gb).resize((sw, sh), Image.Resampling.LANCZOS))
    out = src.copy()
    area = out[sb[1]:sb[3], sb[0]:sb[2]]
    source_area = src[sb[1]:sb[3], sb[0]:sb[2]]
    # Original antialiased edge pixels are retained verbatim. This also avoids
    # expanding partially transparent generated edge colors into opaque pixels.
    reliable = (mapped[:, :, 3] >= alpha_threshold) & (source_area[:, :, 3] == 255)
    area[reliable, :3] = mapped[reliable, :3]
    fallback = (source_area[:, :, 3] == 255) & ~reliable
    protected = np.zeros(src.shape[:2], dtype=bool)
    if protect_mask:
        m = Image.open(protect_mask).convert("L")
        if m.size != src_im.size:
            raise ValueError("Protection mask must have exact source dimensions")
        values = np.array(m)
        if not np.all((values == 0) | (values == 255)):
            raise ValueError("Protection core must be binary; feather only outside it")
        protected = values == 255
        if not protected.any():
            raise ValueError("Protection mask is empty")
        if feather:
            # Protect core stays exact. Optional transition only outside core.
            expanded = m.filter(ImageFilter.MaxFilter(2 * feather + 1))
            weight = np.array(expanded.filter(ImageFilter.GaussianBlur(feather / 2)), dtype=np.float32) / 255
            weight[protected] = 1
            out[:, :, :3] = np.rint(src[:, :, :3] * weight[:, :, None] + out[:, :, :3] * (1-weight[:, :, None])).astype(np.uint8)
        out[protected] = src[protected]
    # Source alpha is the geometric authority; zero-alpha RGB also stays source.
    out[:, :, 3] = src[:, :, 3]
    out[src[:, :, 3] == 0] = src[src[:, :, 3] == 0]
    output = paths[2]
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(output)
    saved = np.array(Image.open(output).convert("RGBA"))
    report = {
        "source": str(paths[0]), "generated": str(paths[1]), "output": str(output),
        "source_sha256": sha256(paths[0]), "generated_sha256": sha256(paths[1]),
        "output_sha256": sha256(output), "source_size": list(src_im.size),
        "generated_size": list(gen_im.size), "output_size": [saved.shape[1], saved.shape[0]],
        "source_box": sb, "generated_box": gb, "aspect_error": aspect_error,
        "scale_xy": [sw/gw, sh/gh], "alignment_visually_confirmed": True,
        "alpha_exact": bool(np.array_equal(saved[:, :, 3], src[:, :, 3])),
        "size_exact": saved.shape == src.shape,
        "protected_pixels": int(protected.sum()),
        "protected_changed_pixels": int(np.any(saved[protected] != src[protected], axis=1).sum()),
        "source_opaque_gap_pixels_retained": int(fallback.sum()),
        "source_antialiased_edge_pixels_retained": int(((src[:, :, 3] > 0) & (src[:, :, 3] < 255)).sum()),
        "protected_mask": str(Path(protect_mask).resolve()) if protect_mask else None,
        "protected_mask_sha256": sha256(protect_mask) if protect_mask else None,
        "visual_fidelity": "requires separate visual review; mechanical checks do not certify defects",
    }
    assert report["alpha_exact"] and report["size_exact"] and report["protected_changed_pixels"] == 0
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--generated", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-box", nargs=4, type=int)
    p.add_argument("--generated-box", nargs=4, type=int)
    p.add_argument("--protect-mask")
    p.add_argument("--feather", type=int, default=0)
    p.add_argument("--alpha-threshold", type=int, default=192)
    p.add_argument("--max-aspect-error", type=float, default=0.06)
    p.add_argument("--confirm-alignment", action="store_true")
    p.add_argument("--report", required=True)
    args = vars(p.parse_args())
    report_path = Path(args.pop("report"))
    if report_path.exists() or report_path.resolve() in {Path(args[x]).resolve() for x in ("source", "generated", "output")}:
        p.error("Report must be a new separate file")
    if args["feather"] < 0 or args["feather"] > 32 or not 1 <= args["alpha_threshold"] <= 255:
        p.error("Invalid feather/alpha threshold")
    try:
        report = finalize(**args)
    except ValueError as e:
        p.error(str(e))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
