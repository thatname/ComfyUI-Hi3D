"""Boundary-loop detection and ear-clipping hole filling for extracted meshes."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from sparc3d.structures.point import Mesh


def fill_mesh_holes(
    mesh: Mesh,
    length_thresh: int = 10000,
    device: torch.device | str | None = None,
) -> Mesh:
    """Close every boundary loop of a triangle mesh by ear clipping.

    Boundary half-edges are found on the GPU, then the loops are triangulated on the
    CPU with numpy, all loops of one length at a time. Loops longer than
    ``length_thresh`` are left open, since ear clipping is quadratic in the loop
    length.

    Args:
        mesh: Mesh to repair. Vertices are reused unchanged; only faces are added.
        length_thresh: Maximum boundary-loop length to fill.
        device: Device for boundary detection; defaults to the mesh's own, and
            lets a host-resident mesh use a GPU.

    Returns:
        New mesh with the fill triangles appended.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    work = faces.device if device is None else torch.device(device)
    try:
        boundary_directed = _boundary_half_edges(faces, vertices.shape[0], work)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        boundary_directed = _boundary_half_edges(faces, vertices.shape[0], torch.device("cpu"))

    loops = _boundary_loops(boundary_directed)
    loops = [loop for loop in loops if 3 <= len(loop) <= length_thresh]
    if loops:
        positions = vertices.detach().cpu().numpy()
        appended = torch.from_numpy(_ear_clip_loops(positions, loops)).long()
        faces = torch.cat([faces, appended.to(faces.device)], dim=0)

    return Mesh(vertices=vertices, faces=faces)


def _boundary_half_edges(
    faces: torch.Tensor, num_vertices: int, device: torch.device
) -> np.ndarray:
    """Directed half-edges ``[B, 2]`` whose undirected edge occurs once.

    Half-edges are numbered as ``[v0→v1 of every face, v1→v2, v2→v0]`` and returned
    in that order. Sorting one int64 key per half-edge keeps the working set to a
    few copies of that key, where a row-wise ``torch.unique`` needs several times
    more on meshes of tens of millions of faces.
    """
    face_count = faces.shape[0]
    # Keys, the sorted copy, its order and the sort's scratch: about six int64 per edge.
    if device.type == "cuda" and torch.cuda.mem_get_info(device)[0] < 3 * face_count * 8 * 6:
        device = torch.device("cpu")
    faces = faces.to(device).int()
    starts = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]])
    ends = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0]])
    keys = torch.minimum(starts, ends).long() * num_vertices + torch.maximum(starts, ends).long()
    del starts, ends
    keys, order = torch.sort(keys)
    distinct = torch.ones(keys.shape[0] + 1, dtype=torch.bool, device=device)
    distinct[1:-1] = keys[1:] != keys[:-1]
    del keys
    single = distinct[:-1] & distinct[1:]
    half_edges = torch.sort(order[single]).values
    del order, distinct, single
    slot, face = half_edges // face_count, half_edges % face_count
    boundary = torch.stack([faces[face, slot], faces[face, (slot + 1) % 3]], dim=1)
    return boundary.cpu().numpy().astype(int)


def _ear_clip_loops(positions: np.ndarray, loops: List[List[int]]) -> np.ndarray:
    """Triangulate many boundary loops as :func:`_ear_clip` does, batched by length.

    Every step clips one ear from each loop of the current longest length, which
    then joins the loops one vertex shorter, so the whole set takes as many
    vectorized steps as the longest loop has vertices. The arithmetic is
    :func:`_ear_clip`'s, and the triangles are returned in the order loop-by-loop
    clipping would emit them.
    """
    lengths = np.array([len(loop) for loop in loops])
    offsets = np.zeros(len(loops) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths - 2)
    out = np.empty((int(offsets[-1]), 3), dtype=int)

    # length -> (loop ids, vertex indices [B, length], positions [B, length, 3])
    groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for length in np.unique(lengths):
        members = np.nonzero(lengths == length)[0]
        idx = np.array([loops[i] for i in members], dtype=int)
        groups[int(length)] = (members, idx, positions[idx])

    while groups:
        current = max(groups)
        members, idx, boundline = groups.pop(current)
        written = lengths[members] - current
        batch = np.arange(len(members))
        if current == 3:
            out[offsets[members] + written] = idx[:, [2, 1, 0]]
            continue
        next_pts = np.roll(boundline, -1, axis=1)
        T = next_pts - boundline
        T = T / (np.linalg.norm(T, axis=2, keepdims=True) + 1e-6)
        T_prev = np.roll(T, 1, axis=1)
        N = np.cross(-T_prev, T)
        center = boundline.mean(axis=1, keepdims=True)
        P = next_pts - center
        Q = boundline - center
        sin_sign = np.sign((np.cross(P, Q) * N).sum(axis=2, keepdims=True))
        sin_val = np.linalg.norm(N, axis=2, keepdims=True) * sin_sign
        cos_val = -(T_prev * T).sum(axis=2, keepdims=True)
        A = np.arctan2(sin_val, cos_val)[..., 0]
        collapsed = A == 0.0
        score = np.where(collapsed, np.inf, 1.0 / np.where(collapsed, 1.0, A))
        ear = np.argmax(score, axis=1)
        i0 = (ear + 1) % current
        i2 = (ear - 1) % current
        out[offsets[members] + written] = np.stack(
            [idx[batch, i0], idx[batch, ear], idx[batch, i2]], axis=1
        )
        keep = np.ones(idx.shape, dtype=bool)
        keep[batch, ear] = False
        shorter = (
            members,
            idx[keep].reshape(len(members), current - 1),
            boundline[keep].reshape(len(members), current - 1, 3),
        )
        if current - 1 in groups:
            shorter = tuple(
                np.concatenate([a, b]) for a, b in zip(groups[current - 1], shorter)
            )
        groups[current - 1] = shorter
    return out


def _boundary_loops(boundary: np.ndarray) -> List[List[int]]:
    """Chain boundary half-edges ``[B, 2]`` into closed vertex loops.

    Half-edges are consumed at most once, so loops that share a vertex are still
    separated instead of merging into one traversal.
    """
    outgoing: dict[int, List[int]] = {}
    for idx, (start, _) in enumerate(boundary):
        outgoing.setdefault(start, []).append(idx)
    visited = [False] * len(boundary)

    loops: List[List[int]] = []
    for i in range(len(boundary)):
        if visited[i]:
            continue
        visited[i] = True
        start_v = boundary[i][0]
        loop = [start_v, boundary[i][1]]

        while loop[-1] != start_v:
            for next_idx in outgoing.get(loop[-1], []):
                if not visited[next_idx]:
                    visited[next_idx] = True
                    loop.append(boundary[next_idx][1])
                    break
            else:
                break  # dead end: not a closed loop

        if loop[0] == loop[-1] and len(loop) > 3:
            loop.pop()  # drop the duplicated start vertex
            loops.append(loop)
    return loops


def _ear_clip(vertices: torch.Tensor, boundary: List[int]) -> np.ndarray:
    """Triangulate one boundary loop, repeatedly clipping its sharpest corner."""
    boundline = vertices[torch.tensor(boundary, device=vertices.device)].cpu().numpy()
    boundline_idx = np.array(boundary, dtype=int)

    faces_hole = []
    while True:
        if len(boundline_idx) == 3:
            faces_hole.append(boundline_idx[[2, 1, 0]])
            break

        next_pts = np.roll(boundline, -1, axis=0)
        T = next_pts - boundline
        T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-6)
        T_prev = np.roll(T, 1, axis=0)
        N = np.cross(-T_prev, T)

        # Signed interior angle at each corner, relative to the loop centroid.
        center = boundline.mean(axis=0, keepdims=True)
        P = next_pts - center
        Q = boundline - center
        sin_sign = np.sign((np.cross(P, Q) * N).sum(axis=1, keepdims=True))
        sin_val = np.linalg.norm(N, axis=1, keepdims=True) * sin_sign
        cos_val = -(T_prev * T).sum(axis=1, keepdims=True)
        A = np.arctan2(sin_val, cos_val).reshape(-1)

        # Clip the sharpest corner: 1/A ranks the smallest positive angle first and
        # every reflex corner last. Scoring exp(1/A) instead ranks corners the same
        # way, but a folded corner has exactly antiparallel edges, hence A of exactly
        # 0, and dividing by it is fatal rather than merely a warning wherever
        # floating-point traps are unmasked. Such a corner is clipped first, which is
        # where the ranking wants it. Ties among comparably sharp corners can break
        # differently than under exp, which rounds large scores together.
        collapsed = A == 0.0
        score = np.where(collapsed, np.inf, 1.0 / np.where(collapsed, 1.0, A))
        ear = int(np.argmax(score))

        i0 = (ear + 1) % len(boundline_idx)
        i2 = (ear - 1) % len(boundline_idx)
        faces_hole.append(boundline_idx[[i0, ear, i2]])

        boundline = np.delete(boundline, ear, axis=0)
        boundline_idx = np.delete(boundline_idx, ear)

    return np.vstack(faces_hole)
