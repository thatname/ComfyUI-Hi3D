"""Out-of-core, multi-device execution of the decoder's high-resolution stages.

After the attention stack, every decoder layer is a group norm, an activation, a
3x3x3 submanifold convolution, a per-voxel linear map or an occupancy-pruned
upsampling. All of them are local except group norm, whose statistics span the whole
point set. :class:`StreamedDecoder` therefore runs the stages one convolution at a
time: level tensors live in host memory, each pass visits Morton-ordered chunks of
tiles plus a one-voxel halo on any of several GPUs, and group-norm statistics are
accumulated while a tensor is produced and applied when it is consumed. The result
matches :meth:`PTv3MeshDecoder.forward` up to floating-point summation order, while
device memory is bounded by the chunk size instead of the point count.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import torch

from sparc3d.models.autoencoders.blocks import (
    ConvOut,
    PointGroupNorm,
    SparseResnetBlock,
    Upsample,
)
from sparc3d.ops import sparse_backend
from sparc3d.ops.tiling import Chunk, TiledLevel, host_empty, morton_key, run_chunks, TILE_SIZE
from sparc3d.structures.point import Point

DEFAULT_CHUNK_BYTES = 3 * 2**30
"""Approximate device working set of one chunk."""

_CONV_HALO = 1

logger = logging.getLogger(__name__)


@dataclass
class StreamedDecoding:
    """Decoder output held in host memory, in the tile order of ``level``.

    ``feat`` holds the activated per-corner SDF and deformation channels and
    ``sdf_sign`` the per-corner sign logits, exactly as a decoded :class:`Point`.
    """

    level: TiledLevel
    feat: torch.Tensor
    sdf_sign: torch.Tensor
    matrix: torch.Tensor
    layouts: Dict[str, Any]

    @property
    def resolution(self) -> int:
        return self.level.resolution

    @property
    def grid_coord(self) -> torch.Tensor:
        return self.level.coords


class _Stats:
    """Group-norm moments of a level tensor, accumulated chunk by chunk."""

    def __init__(self, channels: int):
        self.sum = torch.zeros(channels, dtype=torch.float64)
        self.sum_sq = torch.zeros(channels, dtype=torch.float64)
        self.count = 0
        self._lock = threading.Lock()

    def add(self, feat: torch.Tensor) -> None:
        feat = feat.float()
        partial = feat.sum(0).double().cpu()
        partial_sq = feat.square().sum(0).double().cpu()
        with self._lock:
            self.sum += partial
            self.sum_sq += partial_sq
            self.count += feat.shape[0]

    def moments(self, norm: PointGroupNorm, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-group ``(mean, inv_std)`` with :class:`PointGroupNorm`'s formula."""
        groups, per_group = norm.group, norm.channels_per_group
        sums = self.sum.view(groups, per_group).sum(1)
        sums_sq = self.sum_sq.view(groups, per_group).sum(1)
        counts = float(self.count * per_group)
        mean = sums / (counts + norm.eps)
        var = sums_sq / (counts + norm.eps) - mean.square()
        inv_std = torch.rsqrt(var + norm.eps)
        return mean.float().to(device), inv_std.float().to(device)


def _group_norm(feat: torch.Tensor, norm: PointGroupNorm, moments) -> torch.Tensor:
    """Normalize with global moments; float32 result, as inside ``force_fp32``."""
    mean, inv_std = moments
    n, channels = feat.shape
    grouped = feat.float().view(n, norm.group, norm.channels_per_group)
    out = (grouped - mean.view(1, -1, 1)) * inv_std.view(1, -1, 1)
    return out.view(n, channels).mul_(norm.weights).add_(norm.bias)


def _swish(feat: torch.Tensor) -> torch.Tensor:
    return feat * torch.sigmoid(feat)


class StreamedDecoder:
    """Run a :class:`PTv3MeshDecoder` with its upsampling stages streamed.

    Args:
        decoder: The decoder, already on ``devices[0]``.
        devices: CUDA devices sharing the chunks; the first also runs the latent
            level and attention stack.
        chunk_bytes: Approximate device working set per chunk.
    """

    def __init__(
        self,
        decoder: Any,
        devices: Sequence[torch.device],
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ):
        self.devices = [torch.device(device) for device in devices]
        self.chunk_bytes = int(chunk_bytes)
        self.decoder = decoder
        self._replicas: Dict[torch.device, Any] = {}

    def replica(self, device: torch.device) -> Any:
        """Return the decoder's copy on ``device``, made once and reused."""
        primary = next(self.decoder.parameters()).device
        if device == primary:
            return self.decoder
        if device not in self._replicas:
            # Only the streamed stages are replicated; the attention stack holds the
            # flash-attn module, which cannot be copied, and runs on the primary only.
            stages = torch.nn.ModuleDict(
                {"dec": self.decoder.dec, "conv_out": self.decoder.conv_out}
            )
            self._replicas[device] = copy.deepcopy(stages).to(device)
        return self._replicas[device]

    def release(self) -> None:
        """Drop the per-device decoder copies."""
        self._replicas.clear()

    # -- entry point ------------------------------------------------------------------

    @torch.inference_mode()
    def decode(self, mapping: Mapping[str, Any]) -> StreamedDecoding:
        """Decode a latent mapping as :meth:`SparconvVAE.decode` does."""
        primary = self.devices[0]
        point = Point(dict(mapping))
        point.setdefault("sdf", None)
        point.setdefault("deform", None)
        point.initialization()
        layouts = {}
        start = 0
        for key in ("sdf", "deform"):
            entry = Point.ALL_LAYOUTS[key]
            layouts[key] = {**entry, "range": (start, start + entry["size"])}
            start += entry["size"]
        point["layouts"] = layouts

        decoder = self.decoder
        with torch.cuda.device(primary), torch.autocast("cuda"):
            point.sparsify()
            point = decoder.z_to_f(point)
            point.serialization(order=decoder.order)
            point = decoder.mid(point)

        level, order = TiledLevel.from_coords(point.grid_coord, point.resolution)
        feat = host_empty(point.feat.shape, point.feat.dtype)
        feat.copy_(point.feat[order.to(point.feat.device)])
        stats = _Stats(feat.shape[1])
        stats.add(point.feat)
        matrix = point.matrix.cpu()
        del point

        for stage_index, stage in enumerate(decoder.dec._modules.values()):
            for name, module in stage._modules.items():
                path = (stage_index, name)
                started = time.perf_counter()
                if isinstance(module, SparseResnetBlock):
                    feat, stats = self._resnet_block(level, feat, stats, path)
                elif isinstance(module, Upsample):
                    level, feat, stats = self._upsample(level, feat, stats, path)
                else:
                    raise TypeError(f"unsupported decoder stage module {type(module).__name__}")
                logger.debug(
                    "streamed dec%d.%s: %d points x %d channels, %.2fs",
                    stage_index, name, level.num_points, feat.shape[1],
                    time.perf_counter() - started,
                )
        sdf_sign, out = self._conv_out(level, feat, stats)
        return StreamedDecoding(level, out, sdf_sign, matrix, layouts)

    # -- helpers ----------------------------------------------------------------------

    def _module(self, device: torch.device, path) -> Any:
        stage_index, name = path
        stage = list(self.replica(device).dec._modules.values())[stage_index]
        return stage._modules[name]

    def _max_points(self, channels: int) -> int:
        per_point = 26 * channels + 400
        return max(50_000, self.chunk_bytes // per_point)

    @staticmethod
    def _gather(tensor: torch.Tensor, chunk: Chunk, device: torch.device) -> torch.Tensor:
        own = tensor[chunk.start:chunk.stop].to(device, non_blocking=True)
        if chunk.halo.numel() == 0:
            return own
        halo = tensor.index_select(0, chunk.halo).to(device, non_blocking=True)
        return torch.cat([own, halo])

    def _sparse(self, level: TiledLevel, chunk: Chunk, feat: torch.Tensor, device) -> Any:
        coords = self._gather(level.coords, chunk, device)
        indices = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
        return sparse_backend.sparse_tensor(feat, indices, spatial_shape=(level.resolution,) * 3)

    def _conv(self, conv, level, chunk, feat, device) -> torch.Tensor:
        out = conv(self._sparse(level, chunk, feat, device))
        return sparse_backend.get_feats(out)[: chunk.num_own]

    def _chunks(self, level: TiledLevel, channels: int) -> List[Chunk]:
        return level.plan_chunks(self._max_points(channels), _CONV_HALO, self.devices[0])

    # -- stages -----------------------------------------------------------------------

    def _resnet_block(self, level, feat, stats, path):
        block = self._module(self.devices[0], path)
        if level.num_points > block.skip_points:
            warnings.warn(
                f"SparseResnetBlock skipped: {level.num_points} points exceeds the "
                f"skip_points budget of {block.skip_points}. The block is a no-op, so "
                "the output is the input; raise skip_points to run it.",
                stacklevel=2,
            )
            return feat, stats

        norm1, _, conv1, norm2, _, conv2 = block.block._modules.values()
        dtype = feat.dtype
        hidden = host_empty((level.num_points, conv1.out_channels), dtype)
        hidden_stats = _Stats(conv1.out_channels)
        chunks = self._chunks(level, max(feat.shape[1], conv1.out_channels))
        moments = {device: stats.moments(norm1, device) for device in self.devices}

        def first(chunk: Chunk, device: torch.device) -> None:
            norm, _, conv, *_ = self._module(device, path).block._modules.values()
            x = self._gather(feat, chunk, device)
            h = _swish(_group_norm(x, norm, moments[device]).to(dtype))
            h = self._conv(conv, level, chunk, h, device)
            hidden_stats.add(h)
            hidden[chunk.start:chunk.stop].copy_(h)

        run_chunks(chunks, self.devices, first)

        out_dtype = dtype
        if block.res_conv is not None:
            with torch.cuda.device(self.devices[0]), torch.autocast("cuda"):
                probe = block.res_conv(torch.zeros(1, feat.shape[1], device=self.devices[0], dtype=dtype))
            out_dtype = torch.promote_types(probe.dtype, dtype)
        in_place = out_dtype == dtype and conv2.out_channels == feat.shape[1]
        out = feat if in_place else host_empty((level.num_points, conv2.out_channels), out_dtype)
        out_stats = _Stats(conv2.out_channels)
        moments = {device: hidden_stats.moments(norm2, device) for device in self.devices}

        def second(chunk: Chunk, device: torch.device) -> None:
            module = self._module(device, path)
            *_, norm, _, conv = module.block._modules.values()
            h = self._gather(hidden, chunk, device)
            h = _swish(_group_norm(h, norm, moments[device]).to(dtype))
            h = self._conv(conv, level, chunk, h, device)
            x = feat[chunk.start:chunk.stop].to(device, non_blocking=True)
            with torch.autocast("cuda"):
                shortcut = module.res_conv(x) if module.res_conv is not None else x
                y = shortcut + h
            out_stats.add(y)
            out[chunk.start:chunk.stop].copy_(y)

        run_chunks(chunks, self.devices, second)
        return out, out_stats

    def _upsample(self, level, feat, stats, path):
        unpool = self._module(self.devices[0], path).upsample
        norm = next(m for m in unpool.cls_head.fc._modules.values() if isinstance(m, PointGroupNorm))
        chunks = self._chunks(level, feat.shape[1] * 4)
        masks = host_empty((level.num_points, 8), torch.bool)
        moments = {device: stats.moments(norm, device) for device in self.devices}

        def occupancy(chunk: Chunk, device: torch.device) -> None:
            head = self._module(device, path).upsample.cls_head
            gn, _, conv = head.fc._modules.values()
            x = self._gather(feat, chunk, device)
            h = _swish(_group_norm(x, gn, moments[device]))
            logits = self._conv(conv, level, chunk, h, device)
            masks[chunk.start:chunk.stop].copy_(logits > 0)

        run_chunks(chunks, self.devices, occupancy)

        child_counts = torch.tensor([int(masks[c.start:c.stop].sum()) for c in chunks])
        child_offsets = torch.zeros(len(chunks) + 1, dtype=torch.long)
        child_offsets[1:] = torch.cumsum(child_counts, 0)
        total = int(child_offsets[-1])
        channels = unpool.proj.out_features // 8
        with torch.cuda.device(self.devices[0]), torch.autocast("cuda"):
            probe = unpool.proj(torch.zeros(1, feat.shape[1], device=self.devices[0], dtype=feat.dtype))
        child_feat = host_empty((total, channels), probe.dtype)
        child_coords = host_empty((total, 3), torch.int32)
        child_stats = _Stats(channels)
        tile_parts: List[tuple[torch.Tensor, torch.Tensor]] = [None] * len(chunks)

        def split(chunk: Chunk, device: torch.device) -> None:
            module = self._module(device, path).upsample
            x = feat[chunk.start:chunk.stop].to(device, non_blocking=True)
            mask = masks[chunk.start:chunk.stop].to(device).flatten()
            with torch.autocast("cuda"):
                children = module.proj(x)
            children = children.view(-1, channels)[mask]
            coords = level.coords[chunk.start:chunk.stop].to(device)
            coords = (coords.unsqueeze(1) * 2 + module.inv_grid_coord.unsqueeze(0).int()).view(-1, 3)[mask]
            keys = morton_key(torch.div(coords, TILE_SIZE, rounding_mode="floor"))
            keys, order = torch.sort(keys, stable=True)
            children, coords = children[order], coords[order]
            child_stats.add(children)
            lo, hi = int(child_offsets[chunk.index]), int(child_offsets[chunk.index + 1])
            child_feat[lo:hi].copy_(children)
            child_coords[lo:hi].copy_(coords)
            unique, counts = torch.unique_consecutive(keys, return_counts=True)
            tile_parts[chunk.index] = (unique.cpu(), counts.cpu())

        run_chunks(chunks, self.devices, split)
        del masks
        tile_keys = torch.cat([part[0] for part in tile_parts])
        counts = torch.cat([part[1] for part in tile_parts])
        tile_start = torch.zeros(tile_keys.numel() + 1, dtype=torch.long)
        tile_start[1:] = torch.cumsum(counts, 0)
        child = TiledLevel(child_coords, level.resolution * 2, tile_keys, tile_start)
        return child, child_feat, child_stats

    def _conv_out(self, level, feat, stats):
        head: ConvOut = self.decoder.conv_out
        norm = next(m for m in head.conv._modules.values() if isinstance(m, PointGroupNorm))
        chunks = self._chunks(level, feat.shape[1])
        out_channels = next(
            m for m in head.conv._modules.values() if sparse_backend.is_sparse_conv(m)
        ).out_channels
        sdf_sign = host_empty((level.num_points, 8), torch.float32)
        out = host_empty((level.num_points, out_channels - 8), torch.float32)
        moments = {device: stats.moments(norm, device) for device in self.devices}
        layouts = {
            "sdf": Point.ALL_LAYOUTS["sdf"],
            "deform": Point.ALL_LAYOUTS["deform"],
        }

        def project(chunk: Chunk, device: torch.device) -> None:
            gn, _, conv = self.replica(device).conv_out.conv._modules.values()
            x = self._gather(feat, chunk, device)
            h = _swish(_group_norm(x, gn, moments[device]))
            h = self._conv(conv, level, chunk, h, device)
            sdf_sign[chunk.start:chunk.stop].copy_(h[:, :8])
            rest = h[:, 8:]
            start = 0
            activated = []
            for entry in layouts.values():
                activated.append(entry["act"](rest[:, start:start + entry["size"]]))
                start += entry["size"]
            out[chunk.start:chunk.stop].copy_(torch.cat(activated, dim=1))

        run_chunks(chunks, self.devices, project)
        return sdf_sign, out
