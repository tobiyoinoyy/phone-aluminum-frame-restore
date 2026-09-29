#!/usr/bin/env python3
"""Read-only UV seam extraction, including geometrically overlapping open edges.

Does not alter a Blender scene or any source texture. Every virtual segment is
the exact overlap of two collinear source edges; no gap is guessed or bridged.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def build(scene, tolerance=1e-6):
    faces = {f['id']: f for f in scene['faces']}
    matrix = np.asarray(scene.get('matrix_world', np.eye(4)), float)
    verts = {v['id']: (matrix @ np.r_[np.asarray(v['co'], float), 1.])[:3]
             for v in scene['vertices']}
    sizes = {}
    for i, material in enumerate(scene['materials']):
        node = next(n for n in material['nodes'] if n['type'] == 'ShaderNodeTexImage')
        with Image.open(os.path.normpath(node['path'])) as image:
            sizes[i] = image.size

    def pixel(face, vertex):
        u, v = face['uv'][face['vertices'].index(vertex)]
        w, h = sizes[face['material']]
        return np.array([u*w-.5, (1-v)*h-.5])

    def pixel_on_edge(face, edge, position):
        a, b = (verts[v] for v in edge['verts'])
        t = np.dot(position-a, b-a) / np.dot(b-a, b-a)
        if t < -tolerance or t > 1+tolerance:
            raise ValueError('Virtual point is outside original edge')
        p, q = (pixel(face, v) for v in edge['verts'])
        return p + t*(q-p)

    grouped = defaultdict(list)

    def append(ea, eb, fa, fb, a, b, kind):
        if fa['material'] > fb['material']:
            ea, eb, fa, fb = eb, ea, fb, fa
        grouped[(fa['material'], fb['material'])].append({
            'edges': [ea, eb], 'faces': [fa, fb],
            'co': [a, b], 'kind': kind,
        })

    for edge in scene['edges']:
        if len(edge['faces']) != 2:
            continue
        fa, fb = (faces[i] for i in edge['faces'])
        if fa['material'] != fb['material']:
            append(edge, edge, fa, fb, *(verts[v] for v in edge['verts']), 'shared')

    open_edges = [e for e in scene['edges'] if len(e['faces']) == 1]
    for i, ea in enumerate(open_edges):
        a, b = (verts[v] for v in ea['verts'])
        length = np.linalg.norm(b-a)
        if length <= tolerance:
            continue
        tangent = (b-a)/length
        fa = faces[ea['faces'][0]]
        for eb in open_edges[i+1:]:
            fb = faces[eb['faces'][0]]
            if fa['material'] == fb['material']:
                continue
            c, d = (verts[v] for v in eb['verts'])
            other_length = np.linalg.norm(d-c)
            if other_length <= tolerance:
                continue
            if np.linalg.norm(np.cross(tangent, (d-c)/other_length)) > 1e-5:
                continue
            error = max(np.linalg.norm(np.cross(c-a, tangent)),
                        np.linalg.norm(np.cross(d-a, tangent)))
            if error > tolerance:
                continue
            interval = sorted([np.dot(c-a, tangent), np.dot(d-a, tangent)])
            lo, hi = max(0., interval[0]), min(length, interval[1])
            if hi-lo > tolerance:
                append(ea, eb, fa, fb, a+tangent*lo, a+tangent*hi, 'open_overlap')

    results, diagnostics = [], []
    for mats, segments in sorted(grouped.items()):
        nodes, adjacency = [], defaultdict(list)

        def node_id(position):
            for i, existing in enumerate(nodes):
                if np.linalg.norm(position-existing) <= tolerance:
                    return i
            nodes.append(position.copy())
            return len(nodes)-1

        for i, segment in enumerate(segments):
            segment['nodes'] = [node_id(p) for p in segment['co']]
            for vertex in segment['nodes']:
                adjacency[vertex].append(i)
        if any(len(es) > 2 for es in adjacency.values()):
            raise ValueError('Branching or duplicated geometric seam segments')
        remaining = set(range(len(segments)))
        while remaining:
            endpoints = [v for v, es in adjacency.items()
                         if sum(e in remaining for e in es) == 1]
            if not endpoints:
                raise ValueError('Closed seam requires explicit parameter origin')
            current = min(endpoints, key=lambda v: tuple(nodes[v]))
            order, ordered = [current], []
            while True:
                candidates = [e for e in adjacency[current] if e in remaining]
                if not candidates:
                    break
                index = candidates[0]
                ordered.append(segments[index])
                remaining.remove(index)
                current = next(v for v in segments[index]['nodes'] if v != current)
                order.append(current)
            co = np.array([nodes[v] for v in order])
            lengths = np.linalg.norm(np.diff(co, axis=0), axis=1)
            arc = np.r_[0, np.cumsum(lengths)]
            sides, maximum_uv_jump = [], 0.
            for j, mat in enumerate(mats):
                points, normals, ratios = [], [], []
                for i, segment in enumerate(ordered):
                    face, edge = segment['faces'][j], segment['edges'][j]
                    p, q = (pixel_on_edge(face, edge, position)
                            for position in [co[i], co[i+1]])
                    if i:
                        gap = np.linalg.norm(points[-1]-p)
                        maximum_uv_jump = max(maximum_uv_jump, float(gap))
                        if gap > .01:
                            raise ValueError('UV is discontinuous within one material chain')
                    else:
                        points.append(p)
                    points.append(q)
                    t = (q-p)/np.linalg.norm(q-p)
                    n = np.array([-t[1], t[0]])
                    uv_centroid = np.mean([pixel(face, v) for v in face['vertices']], axis=0)
                    offset = uv_centroid-(p+q)/2
                    if np.dot(offset, n) < 0:
                        n = -n
                    normals.append(n)
                    world_centroid = np.mean([verts[v] for v in face['vertices']], axis=0)
                    world_offset = world_centroid-(co[i]+co[i+1])/2
                    world_t = (co[i+1]-co[i])/lengths[i]
                    world_offset -= np.dot(world_offset, world_t)*world_t
                    ratios.append(abs(np.dot(offset, n))/np.linalg.norm(world_offset))
                normals = np.asarray(normals)
                vertex_normals = np.vstack([normals[0], normals[:-1]+normals[1:], normals[-1]])
                vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, None]
                points = np.asarray(points)
                sides.append({
                    'name': scene['materials'][mat]['name'],
                    'points': points.tolist(), 'normals': vertex_normals.tolist(),
                    'px_per_unit': float(np.median(ratios)),
                    'segment_px_per_unit': ratios,
                    'source_edge_ids': [s['edges'][j]['id'] for s in ordered],
                    'source_face_ids': [s['faces'][j]['id'] for s in ordered],
                    'image_size': list(sizes[mat]),
                    'arc_pixels': float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
                })
            shared_length = sum(lengths[i] for i, s in enumerate(ordered) if s['kind'] == 'shared')
            result = {
                'names': [s['name'] for s in sides], 'arc': arc.tolist(), 'sides': sides,
                'edges': [str(s['edges'][0]['id']) if s['kind'] == 'shared'
                          else str(s['edges'][0]['id'])+':'+str(s['edges'][1]['id'])
                          for s in ordered],
                'world_points': co.tolist(),
                'segment_kinds': [s['kind'] for s in ordered],
                'edge_id_pairs': [[s['edges'][0]['id'], s['edges'][1]['id']] for s in ordered],
            }
            results.append(result)
            diagnostics.append({
                'pair': result['names'], 'segments': len(ordered),
                'shared_segments': sum(s['kind'] == 'shared' for s in ordered),
                'open_overlap_segments': sum(s['kind'] == 'open_overlap' for s in ordered),
                'shared_arc_units': float(shared_length), 'complete_arc_units': float(arc[-1]),
                'added_arc_units': float(arc[-1]-shared_length),
                'maximum_within_material_uv_join_error_pixels': maximum_uv_jump,
                'bounds_pixels': {s['name']: [np.min(s['points'], axis=0).tolist(),
                                              np.max(s['points'], axis=0).tolist()] for s in sides},
                'along_seam_pixel_length_ratio': sides[0]['arc_pixels']/sides[1]['arc_pixels'],
            })
    return {'schema': 'paired-uv-seam-chains-v1', 'tolerance_world_units': tolerance,
            'chains': results, 'diagnostics': diagnostics}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = build(json.loads(Path(args.scene).read_text()))
    result['source_scene'] = str(Path(args.scene).resolve())
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result['diagnostics'], ensure_ascii=False, indent=2))
