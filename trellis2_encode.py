"""Encode a mesh into a ComfyUI-native Trellis2 shape latent.

ComfyUI ships the Trellis2 shape decoder but not its encoder, so a mesh made outside
the Trellis2 cascade (here, Hi3D's refinement) cannot condition the native texture
stage. This module rebuilds the TRELLIS.2 ``FlexiDualGridVaeEncoder`` on ComfyUI's own
sparse-convolution modules, loads the official ``shape_enc_next_dc_f16c32_fp16``
weights, and emits the same latent dict ``Trellis2UpsampleStage`` produces, plus the
shape subdivisions ``VaeDecodeTextureTrellis`` needs.
"""

import glob
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.latent_formats
import comfy.model_management
import comfy.utils
import folder_paths
from comfy.ldm.trellis2.vae import SparseConv3d, SparseSpatial2Channel, SparseTensor
from comfy_extras.nodes_save_3d import get_mesh_batch_item

logger = logging.getLogger(__name__)

ENCODER_GLOBS = (
    os.path.join("microsoft", "*", "ckpts", "*shape_enc*.safetensors"),
    os.path.join("vae", "*shape_enc*.safetensors"),
    os.path.join("trellis2", "*shape_enc*.safetensors"),
)
"""Where the TRELLIS.2 shape encoder is looked for, relative to the models directory.
The ComfyUI-Trellis2 wrapper downloads it to ``microsoft/TRELLIS.2-4B/ckpts``."""

LEVELS = 4
"""Downsampling stages between the dual grid and the latent (1024 -> 64, 1536 -> 96)."""

_shape_slat_format = comfy.latent_formats.Trellis2ShapeSLAT()
_encoders: dict = {}


CHUNK_ROWS = 1 << 19
"""Voxels per row-wise step. A 1536 dual grid of a detailed mesh has tens of millions of
voxels, so the per-voxel norms and MLPs run over row chunks, writing into place."""


def _rows(feats, fn, out=None):
    """Apply ``fn`` to ``feats`` chunk by chunk, into ``out`` (default: in place)."""
    out = feats if out is None else out
    for start in range(0, feats.shape[0], CHUNK_ROWS):
        out[start:start + CHUNK_ROWS] = fn(feats[start:start + CHUNK_ROWS])
    return out


def _drop_scale_cache(tensor, scale):
    """Forget a finer level's neighbour maps, which a downsampled tensor would carry along."""
    tensor._spatial_cache.pop(str(scale), None)
    return tensor


def _release(tensor):
    """Free a sparse tensor's features and caches while the caller still holds the object."""
    tensor.data["feats"] = None
    tensor._spatial_cache.clear()


class LayerNorm32(nn.LayerNorm):
    """LayerNorm computed in float32, as in the TRELLIS.2 reference modules."""

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, mlp_ratio=4.0):
        super().__init__()
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = SparseConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x):
        h = self.conv(x)
        feats = h.feats
        for start in range(0, feats.shape[0], CHUNK_ROWS):
            rows = slice(start, start + CHUNK_ROWS)
            feats[rows] = self.mlp(self.norm(feats[rows])) + x.feats[rows]
        return h


class ResBlockS2C(nn.Module):
    """Residual block halving the grid by moving each 2x2x2 cell into channels."""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = SparseConv3d(channels, out_channels // 8, 3)
        self.conv2 = SparseConv3d(out_channels, out_channels, 3)
        self.updown = SparseSpatial2Channel(2)

    def forward(self, x):
        """Downsample ``x``, consuming it: its features are normalised in place and freed."""
        # The reference skip moves x into channels (8x the features) and then averages
        # groups of 8 * channels / out_channels consecutive channels. Each group lies in one
        # child slot, so averaging first and moving the result is the same and 8x smaller.
        group = self.channels * 8 // self.out_channels
        pooled = x.replace(x.feats.reshape(x.feats.shape[0], -1, group).mean(dim=-1))

        h = x.replace(_rows(x.feats, lambda f: F.silu(self.norm1(f))))
        h = self.conv1(h)
        scale = x._scale
        _release(x)
        h = _drop_scale_cache(self.updown(h), scale)
        skip = _drop_scale_cache(self.updown(pooled), scale)
        del pooled
        h = h.replace(_rows(h.feats, lambda f: F.silu(self.norm2(f))))
        h = self.conv2(h)
        h.feats.add_(skip.feats)
        return h


class FlexiDualGridEncoder(nn.Module):
    """TRELLIS.2 ``FlexiDualGridVaeEncoder`` (shape_enc_next_dc_f16c32) on ComfyUI sparse ops."""

    def __init__(self, model_channels=(64, 128, 256, 512, 1024), latent_channels=32,
                 num_blocks=(0, 4, 8, 16, 4)):
        super().__init__()
        self.input_layer = nn.Linear(6, model_channels[0])
        self.to_latent = nn.Linear(model_channels[-1], 2 * latent_channels)
        self.blocks = nn.ModuleList()
        for i, count in enumerate(num_blocks):
            stage = nn.ModuleList(ConvNeXtBlock(model_channels[i]) for _ in range(count))
            if i < len(num_blocks) - 1:
                stage.append(ResBlockS2C(model_channels[i], model_channels[i + 1]))
                if count == 0:
                    # The level's only convolution: a cached 27-neighbour map would not be reused.
                    stage[-1].conv1.cache_neighbor_map = False
            self.blocks.append(stage)

    def forward(self, x, dtype):
        """Posterior mean for ``x``: dual-vertex offsets in [0, 1] and per-axis edge flags (6 channels)."""
        h = x.replace(_rows(x.feats, lambda f: self.input_layer(f - 0.5).to(dtype),
                            out=x.feats.new_empty(x.feats.shape[0], self.input_layer.out_features, dtype=dtype)))
        _release(x)
        for stage in self.blocks:
            for block in stage:
                h = block(h)
        feats = F.layer_norm(h.feats.float(), h.feats.shape[-1:])
        mean, _ = self.to_latent(feats).chunk(2, dim=-1)
        return h.replace(mean)


def encoder_files():
    found = []
    for pattern in ENCODER_GLOBS:
        for path in sorted(glob.glob(os.path.join(folder_paths.models_dir, pattern))):
            found.append(os.path.relpath(path, folder_paths.models_dir).replace(os.sep, "/"))
    return found or ["(download microsoft/TRELLIS.2-4B ckpts/shape_enc_next_dc_f16c32_fp16)"]


def _load_encoder(name):
    """Encoder on the CPU, with the torso in float16 as the checkpoint was trained."""
    if name not in _encoders:
        path = os.path.join(folder_paths.models_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Trellis2 shape encoder not found: {path}")
        model = FlexiDualGridEncoder()
        model.load_state_dict(comfy.utils.load_torch_file(path, safe_load=True))
        model.blocks.half()
        for module in model.modules():
            if isinstance(module, LayerNorm32):
                module.float()
        _encoders[name] = model.eval().requires_grad_(False)
    return _encoders[name]


def _to_model_frame(vertices, model_frame):
    """Undo the frame change VaeDecodeShapeTrellis applies to Z-up (Trellis2) decodes."""
    if model_frame == "z_up":
        return torch.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], dim=-1)
    return vertices


def _grid_keys(coords, size):
    coords = coords.long()
    return (coords[:, 0] * size + coords[:, 1]) * size + coords[:, 2]


def _shape_subdivisions(latent_coords, voxels, resolution):
    """The child-occupancy masks the shape decoder would predict, read off the voxels.

    Level ``k`` holds one ``[N_k, 8]`` mask whose rows follow the order the texture
    decoder expands the latent in: the latent's own order at level 0, then parent by
    parent, child by child. Child ``s`` of a cell sits at ``2 * cell + bits(s)``, bit
    ``i`` of ``s`` being the offset along axis ``i``, as in ``SparseChannel2Spatial``.
    """
    bits = torch.tensor([[(s >> i) & 1 for i in range(3)] for s in range(8)],
                        dtype=torch.int32, device=voxels.device)
    parents = latent_coords.int()
    levels = []
    for level in range(LEVELS):
        shift = LEVELS - level - 1
        size = resolution >> shift
        children = torch.unique(_grid_keys(voxels >> shift, size))
        candidates = (2 * parents[:, None, :] + bits[None]).reshape(-1, 3)
        keys = _grid_keys(candidates, size)
        found = torch.searchsorted(children, keys).clamp_max(children.shape[0] - 1)
        occupied = children[found] == keys
        levels.append((parents, occupied.reshape(-1, 8)))
        parents = candidates[occupied]
    return levels


def encode_mesh(encoder, vertices, faces, resolution, device):
    """Shape latent feats ``[N, 32]``, its coords ``[N, 3]`` and per-stage (cells, masks)."""
    from o_voxel.convert import mesh_to_flexible_dual_grid

    voxels, dual_vertices, intersected = mesh_to_flexible_dual_grid(
        vertices.float().cpu(), faces.int().cpu(),
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
    )
    if voxels.shape[0] == 0:
        raise ValueError("mesh lies outside the [-0.5, 0.5] Trellis2 cube; nothing to encode")
    # Full-resolution features and the first convolution's input and output.
    comfy.model_management.free_memory(2 * 1024 ** 3 + 400 * voxels.shape[0], device)
    voxels = voxels.int()
    feats = torch.cat([dual_vertices * resolution - voxels, intersected.float()], dim=1)
    coords = torch.cat([torch.zeros_like(voxels[:, :1]), voxels], dim=1)
    del dual_vertices, intersected
    latent = encoder(SparseTensor(feats=feats.to(device), coords=coords.to(device)), torch.float16)
    del feats, coords
    levels = _shape_subdivisions(latent.coords[:, 1:], voxels.to(device), resolution)
    return latent.feats.float(), latent.coords[:, 1:].int(), levels


class Trellis2EncodeShapeMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Mesh in the frame of the Trellis2 shape decode it replaces "
                                             "(e.g. Hi3D Refine Mesh with match_input_frame on)."}),
                "encoder": (encoder_files(), {"tooltip": "TRELLIS.2 shape VAE encoder, shape_enc_next_dc_f16c32_fp16."}),
                "resolution": (["1024", "1536"], {"default": "1536",
                                                  "tooltip": "Dual-grid resolution; the latent grid is 1/16 of it "
                                                             "(64 or 96), as after Trellis2UpsampleStage."}),
            },
            "optional": {
                "reference_latent": ("LATENT", {"tooltip": "The shape latent the mesh came from. Supplies the model "
                                                           "frame (Pixal3D Y-up or Trellis2 Z-up); Y-up without it."}),
            },
        }

    RETURN_TYPES = ("LATENT", "SHAPE_SUBDIVIDES")
    RETURN_NAMES = ("shape_latent", "shape_subdivides")
    FUNCTION = "encode"
    CATEGORY = "3d/Hi3D"
    DESCRIPTION = ("Encodes a mesh with the TRELLIS.2 shape VAE encoder into a native Trellis2 shape latent "
                   "for Trellis2 Texture Stage, and the matching shape_subdivides for Trellis2 VAE Decode Texture.")

    def encode(self, mesh, encoder, resolution, reference_latent=None):
        resolution = int(resolution)
        model_frame = (reference_latent or {}).get("model_frame", "y_up")
        device = comfy.model_management.get_torch_device()
        model = _load_encoder(encoder)

        feats_list, coords_list, levels_list = [], [], []
        try:
            model.to(device)
            with torch.inference_mode():
                for i in range(int(mesh.vertices.shape[0])):
                    vertices, faces, *_ = get_mesh_batch_item(mesh, i)
                    vertices = _to_model_frame(vertices.float(), model_frame).clamp(-0.5, 0.5)
                    feats, coords, levels = encode_mesh(model, vertices, faces, resolution, device)
                    feats_list.append(feats.cpu())
                    coords_list.append(coords.cpu())
                    levels_list.append([(cells.cpu(), mask.cpu()) for cells, mask in levels])
                    logger.info("Trellis2 encode: item %d, %d faces -> %d latent tokens at %d (%s)",
                                i, faces.shape[0], feats.shape[0], resolution, model_frame)
        finally:
            model.to("cpu")
            comfy.model_management.soft_empty_cache()

        counts = [f.shape[0] for f in feats_list]
        samples = torch.zeros(len(counts), 32, max(counts), 1)
        for i, feats in enumerate(feats_list):
            samples[i, :, :counts[i], 0] = _shape_slat_format.process_in(feats).T
        coords = torch.cat([torch.cat([torch.full_like(c[:, :1], i), c], dim=1)
                            for i, c in enumerate(coords_list)])

        # One SparseTensor per decoder stage, rows in expansion order, item by item. The
        # decoder thresholds the masks at zero, like the logits a shape decode returns.
        subdivides = []
        for level in range(LEVELS):
            stage = [item[level] for item in levels_list]
            subdivides.append(SparseTensor.from_tensor_list(
                [mask.half() * 2 - 1 for _, mask in stage],
                [torch.cat([cells[:, :1], cells], dim=1) for cells, _ in stage],
            ))

        latent = {
            "samples": samples,
            "coords": coords,
            "coord_counts": torch.tensor(counts, dtype=torch.int64),
            "coord_resolution": resolution // 16,
            "type": "trellis2",
            "model_frame": model_frame,
        }
        return (latent, subdivides)


NODE_CLASS_MAPPINGS = {"Trellis2EncodeShapeMesh": Trellis2EncodeShapeMesh}
NODE_DISPLAY_NAME_MAPPINGS = {"Trellis2EncodeShapeMesh": "Trellis2 Encode Mesh (Shape Latent)"}
