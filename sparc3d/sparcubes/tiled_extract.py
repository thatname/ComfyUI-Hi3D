"""Tiled isosurface extraction from a :class:`StreamedDecoding`.

:func:`sparc3d.sparcubes.extract.point_to_mesh` needs every decoded cube on one
device at once. Here the cubes are processed in the decoder's Morton tile chunks with
a three-cube halo, which makes every quantity a chunk keeps exact:

* corner values are averaged over all cubes sharing the corner, exact for corners
  of cubes up to two rings out;
* ambiguous FlexiCubes cases are resolved against a face neighbour, exact for cubes
  up to one ring out;
* each dual vertex is kept by the chunk owning its cube, and each quadrilateral by
  the chunk owning the cube at the minimum corner of its edge, whose other three
  cubes lie within one ring.

Dual vertices are identified across chunks by ``(global cube index, rank within the
cube)``, so the chunks' faces weld into the same mesh the untiled extractor builds.
"""

from __future__ import annotations

import functools
from typing import List, Sequence

import torch

from sparc3d.ops.flexicubes import FlexiCubes
from sparc3d.ops.tiling import Chunk, run_chunks
from sparc3d.sparcubes.sparse_cube import sparse_cube2verts
from sparc3d.structures.point import Mesh, Point

EXTRACT_HALO = 3
"""Cube rings around a chunk needed for exact corner averaging and case resolution."""

DEFAULT_EXTRACT_POINTS = 4_000_000
"""Own cubes per extraction chunk."""

_MAX_DUAL_VERTICES = 4
"""Largest number of dual vertices FlexiCubes emits for one cube."""


@functools.lru_cache(maxsize=None)
def _extractor(device: torch.device) -> FlexiCubes:
    return FlexiCubes(device=device)


def _grid_key(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    coords = coords.long()
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def _dual_vertices(fc: FlexiCubes, voxelgrid_vertices, surf_edges, scalar_field, case_ids, idx_map):
    """:meth:`FlexiCubes._compute_vd`, also returning each dual vertex's cube."""
    surf_edges_x = torch.index_select(voxelgrid_vertices, 0, surf_edges.reshape(-1)).reshape(-1, 2, 3)
    surf_edges_s = torch.index_select(scalar_field, 0, surf_edges.reshape(-1)).reshape(-1, 2, 1)
    idx_map = idx_map.reshape(-1, 12)
    num_vd = torch.index_select(fc.num_vd_table, 0, case_ids)
    edge_group, edge_group_to_vd, edge_group_to_cube = [], [], []
    total_num_vd = 0
    vd_idx_map = torch.zeros((case_ids.shape[0], 12), dtype=torch.int64, device=fc.device)
    for num in torch.unique(num_vd):
        cur_cubes = num_vd == num
        curr_num_vd = cur_cubes.sum() * num
        curr_edge_group = fc.dmc_table[case_ids[cur_cubes], :num].reshape(-1, num * 7)
        curr_edge_group_to_vd = torch.arange(curr_num_vd, device=fc.device).unsqueeze(-1).repeat(1, 7) + total_num_vd
        total_num_vd += curr_num_vd
        curr_edge_group_to_cube = torch.arange(idx_map.shape[0], device=fc.device)[cur_cubes].unsqueeze(-1).repeat(
            1, num * 7).reshape_as(curr_edge_group)
        curr_mask = curr_edge_group != -1
        edge_group.append(torch.masked_select(curr_edge_group, curr_mask))
        edge_group_to_vd.append(torch.masked_select(curr_edge_group_to_vd.reshape_as(curr_edge_group), curr_mask))
        edge_group_to_cube.append(torch.masked_select(curr_edge_group_to_cube, curr_mask))
    edge_group = torch.cat(edge_group)
    edge_group_to_vd = torch.cat(edge_group_to_vd)
    edge_group_to_cube = torch.cat(edge_group_to_cube)

    vd = torch.zeros((total_num_vd, 3), device=fc.device, dtype=torch.float)
    edges_per_vd = torch.zeros((total_num_vd, 1), device=fc.device, dtype=torch.float)
    idx_group = torch.gather(idx_map.reshape(-1), 0, edge_group_to_cube * 12 + edge_group)
    x_group = torch.index_select(surf_edges_x, 0, idx_group.reshape(-1)).reshape(-1, 2, 3)
    s_group = torch.index_select(surf_edges_s, 0, idx_group.reshape(-1)).reshape(-1, 2, 1)
    ue_group = fc._linear_interp(s_group.float(), x_group)
    edges_per_vd = edges_per_vd.index_add_(0, edge_group_to_vd, torch.ones_like(ue_group[:, :1]))
    vd = vd.index_add_(0, edge_group_to_vd, ue_group) / edges_per_vd

    v_idx = torch.arange(vd.shape[0], dtype=torch.int64, device=fc.device)
    index = edge_group_to_cube * 12 + edge_group
    vd_idx_map = vd_idx_map.reshape(-1).scatter(0, index, v_idx[edge_group_to_vd])

    vd_cube = torch.empty(vd.shape[0], dtype=torch.int64, device=fc.device)
    vd_cube[edge_group_to_vd] = edge_group_to_cube
    return vd, vd_idx_map, vd_cube


def _quads(scalar_field, surf_edges, edge_counts, idx_map, vd_idx_map, surf_edges_mask):
    """:meth:`FlexiCubes._triangulate` before the split, with each quad's edge."""
    group_mask = (edge_counts == 4) & surf_edges_mask
    group = idx_map.reshape(-1)[group_mask]
    vd_idx = vd_idx_map[group_mask]
    edge_indices, indices = torch.sort(group, stable=True)
    quad_vd_idx = vd_idx[indices].reshape(-1, 4)
    quad_edge = edge_indices.reshape(-1, 4)[:, 0]
    s_edges = scalar_field[surf_edges[quad_edge].reshape(-1)].reshape(-1, 2)
    flip_mask = s_edges[:, 0] > 0
    quad_vd_idx = torch.cat((quad_vd_idx[flip_mask][:, [0, 1, 3, 2]], quad_vd_idx[~flip_mask][:, [2, 3, 1, 0]]))
    quad_edge = torch.cat((quad_edge[flip_mask], quad_edge[~flip_mask]))
    return quad_vd_idx, quad_edge


def tiled_point_to_mesh(
    decoding,
    devices: Sequence[torch.device],
    max_points: int = DEFAULT_EXTRACT_POINTS,
) -> Mesh:
    """Extract the mesh of a :class:`StreamedDecoding` chunk by chunk.

    Returns:
        Host mesh equal to :func:`point_to_mesh` on the same decoded field, up to
        vertex order.
    """
    level = decoding.level
    resolution = level.resolution
    chunks = level.plan_chunks(max_points, EXTRACT_HALO, devices[0])
    layout_point = Point({"layouts": decoding.layouts})
    matrix = decoding.matrix

    def extract(chunk: Chunk, device: torch.device):
        rows = chunk.rows().to(device)
        coords = level.coords.index_select(0, chunk.rows()).to(device)
        feat = decoding.feat.index_select(0, chunk.rows()).to(device)
        sign_logits = decoding.sdf_sign.index_select(0, chunk.rows()).to(device)

        order = torch.argsort(_grid_key(coords, resolution))
        rows, coords, feat, sign_logits = rows[order], coords[order], feat[order], sign_logits[order]
        own = (rows >= chunk.start) & (rows < chunk.stop)

        vert_feat, vert_layouts = layout_point.get_vert_feat(feat)
        sdf_sign = torch.sigmoid(sign_logits).unsqueeze(2)
        vert_feat = torch.cat([sdf_sign, vert_feat], dim=2)
        vert_grid, vert_feat, cubes = sparse_cube2verts(coords, vert_feat)
        sdf_sign, vert_feat = vert_feat[:, 0:1], vert_feat[:, 1:]
        sdf = Point.get_attr_from_feat(vert_feat, vert_layouts, "sdf")
        deform = Point.get_attr_from_feat(vert_feat, vert_layouts, "deform")
        vert_coord = Point.grid_to_coord(vert_grid + deform, matrix.to(device))
        sdf = torch.where(sdf_sign >= 0.5, 1, -1) * sdf.abs()

        fc = _extractor(device)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            vert_coord = vert_coord.float()
            sdf = sdf.float()
            surf_cubes, occ_fx8 = fc._identify_surf_cubes(sdf, cubes)
            if surf_cubes.sum() == 0:
                return None
            case_ids = fc._get_case_id_sparse(occ_fx8, surf_cubes, resolution, coords)
            surf_edges, idx_map, edge_counts, surf_edges_mask = fc._identify_surf_edges(sdf, cubes, surf_cubes)
            vd, vd_idx_map, vd_cube = _dual_vertices(fc, vert_coord, surf_edges, sdf, case_ids, idx_map)
            quad_vd, quad_edge = _quads(sdf, surf_edges, edge_counts, idx_map, vd_idx_map, surf_edges_mask)

        surf_index = surf_cubes.nonzero().squeeze(1)
        vd_cube = surf_index[vd_cube]
        first = torch.full((coords.shape[0],), vd.shape[0], dtype=torch.long, device=device)
        first = first.scatter_reduce(0, vd_cube, torch.arange(vd.shape[0], device=device), "amin")
        rank = torch.arange(vd.shape[0], device=device) - first[vd_cube]
        vd_key = rows[vd_cube] * _MAX_DUAL_VERTICES + rank

        # A quad belongs to the chunk owning the cube at its edge's minimum corner.
        edge_ends = vert_grid[surf_edges[quad_edge].long()]
        owner_coord = torch.minimum(edge_ends[:, 0], edge_ends[:, 1])
        own_keys = _grid_key(coords[own], resolution)
        owner_key = _grid_key(owner_coord, resolution)
        position = torch.searchsorted(own_keys, owner_key).clamp(max=own_keys.numel() - 1)
        owned_quad = own_keys[position] == owner_key
        quad_keys = vd_key[quad_vd[owned_quad]]
        triangles = quad_keys[:, fc.quad_split_2.long()].reshape(-1, 3)

        keep = own[vd_cube]
        return vd_key[keep].cpu(), vd[keep].cpu(), triangles.cpu()

    parts = [part for part in run_chunks(chunks, devices, extract) if part is not None]
    if not parts:
        return Mesh(vertices=torch.zeros((0, 3)), faces=torch.zeros((0, 3), dtype=torch.long))
    keys = torch.cat([part[0] for part in parts])
    vertices = torch.cat([part[1] for part in parts])
    triangle_keys = torch.cat([part[2] for part in parts])
    keys, order = torch.sort(keys)
    vertices = vertices[order]
    faces = torch.searchsorted(keys, triangle_keys)
    found = keys[faces.clamp(max=keys.numel() - 1)] == triangle_keys
    if not bool(found.all()):
        raise RuntimeError(
            f"tiled extraction left {int((~found).sum())} face corners without a dual vertex"
        )
    return Mesh(vertices=vertices, faces=faces)
