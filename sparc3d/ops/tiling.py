"""Spatial tiling of sparse voxel sets for out-of-core, multi-device processing.

The high-resolution decoder stages and mesh extraction are local: every operation is
either per voxel or reads a bounded neighbourhood. Partitioning the voxels into
cubic tiles, ordered along a Morton curve, lets consecutive tiles be processed as
*chunks* together with a *halo* of neighbouring voxels, and the chunk's own results
are then exact. Features stay in host memory between passes and each chunk visits a
GPU only while it is being computed, so the decode is bounded by host RAM rather than
by device memory, and chunks can run on several devices at once.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import torch

TILE_SIZE = 32
"""Tile edge in voxels of the level being processed. Must exceed any halo radius."""

_MORTON_BITS = 21


def _spread_bits(value: torch.Tensor) -> torch.Tensor:
    value = value & 0x1FFFFF
    value = (value | value << 32) & 0x1F00000000FFFF
    value = (value | value << 16) & 0x1F0000FF0000FF
    value = (value | value << 8) & 0x100F00F00F00F00F
    value = (value | value << 4) & 0x10C30C30C30C30C3
    value = (value | value << 2) & 0x1249249249249249
    return value


def morton_key(tile_coord: torch.Tensor) -> torch.Tensor:
    """Interleave ``[N, 3]`` non-negative tile coordinates, x most significant.

    With this layout the eight children of tile ``t`` at the next resolution have
    keys ``8 * key(t) + (ox << 2 | oy << 1 | oz)``, so a Morton-ordered parent level
    upsamples into a Morton-ordered child level without a global sort.
    """
    tile_coord = tile_coord.long()
    return (
        _spread_bits(tile_coord[:, 0]) << 2
        | _spread_bits(tile_coord[:, 1]) << 1
        | _spread_bits(tile_coord[:, 2])
    )


@dataclass
class Chunk:
    """Consecutive tiles processed together.

    ``own`` is the half-open row range of the chunk's voxels in level order and
    ``halo`` the rows of other tiles within the halo radius (int64, CPU).
    """

    index: int
    start: int
    stop: int
    halo: torch.Tensor

    @property
    def num_own(self) -> int:
        return self.stop - self.start

    @property
    def num_points(self) -> int:
        return self.num_own + int(self.halo.numel())

    def rows(self) -> torch.Tensor:
        own = torch.arange(self.start, self.stop, dtype=torch.long)
        return torch.cat([own, self.halo])


class TiledLevel:
    """Voxel coordinates of one resolution, stored in Morton tile order.

    Attributes:
        coords: ``[N, 3]`` int32 voxel coordinates on the host, grouped by tile.
        resolution: Grid extent of ``coords``.
        tile_keys: ``[K]`` sorted Morton keys of the occupied tiles.
        tile_start: ``[K + 1]`` row offsets of each tile within ``coords``.
    """

    def __init__(
        self,
        coords: torch.Tensor,
        resolution: int,
        tile_keys: torch.Tensor,
        tile_start: torch.Tensor,
    ):
        self.coords = coords
        self.resolution = int(resolution)
        self.tile_keys = tile_keys
        self.tile_start = tile_start
        self._halo_pairs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def num_points(self) -> int:
        return int(self.coords.shape[0])

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, resolution: int
    ) -> tuple["TiledLevel", torch.Tensor]:
        """Sort arbitrary coordinates into tile order.

        Returns:
            ``(level, order)`` where ``order`` permutes the caller's rows into level
            order (apply it to any per-voxel features).
        """
        coords = coords.int()
        keys = morton_key(torch.div(coords, TILE_SIZE, rounding_mode="floor"))
        keys, order = torch.sort(keys, stable=True)
        tile_keys, counts = torch.unique_consecutive(keys, return_counts=True)
        tile_start = torch.zeros(tile_keys.numel() + 1, dtype=torch.long, device=keys.device)
        tile_start[1:] = torch.cumsum(counts, 0)
        level = cls(
            coords[order].cpu(),
            resolution,
            tile_keys.cpu(),
            tile_start.cpu(),
        )
        return level, order.cpu()

    def halo_pairs(self, radius: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(target_tile_key, row)`` for voxels within ``radius`` of another tile.

        A voxel belongs to a foreign tile's halo when it lies within ``radius`` voxels
        (Chebyshev distance) of that tile's box. Pairs are sorted by target key.
        """
        if radius >= TILE_SIZE:
            raise ValueError(f"halo radius {radius} must be below the tile size {TILE_SIZE}")
        if radius in self._halo_pairs:
            return self._halo_pairs[radius]
        offsets = torch.tensor(
            [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
             if (dx, dy, dz) != (0, 0, 0)],
            dtype=torch.int32,
            device=device,
        )
        tile_keys = self.tile_keys.to(device)
        keys_out, rows_out = [], []
        step = 8_000_000
        for start in range(0, self.num_points, step):
            coords = self.coords[start:start + step].to(device)
            local = coords % TILE_SIZE
            low = local < radius
            high = local >= TILE_SIZE - radius
            boundary = (low | high).any(dim=1).nonzero().squeeze(1)
            if boundary.numel() == 0:
                continue
            low, high = low[boundary], high[boundary]
            tile = torch.div(coords[boundary], TILE_SIZE, rounding_mode="floor")
            for offset in offsets:
                needs_low = offset < 0
                needs_high = offset > 0
                match = ((~needs_low | low) & (~needs_high | high)).all(dim=1)
                if not bool(match.any()):
                    continue
                neighbour = tile[match] + offset
                valid = (neighbour >= 0).all(dim=1)
                neighbour, rows = neighbour[valid], boundary[match][valid]
                keys = morton_key(neighbour)
                position = torch.searchsorted(tile_keys, keys).clamp(max=tile_keys.numel() - 1)
                exists = tile_keys[position] == keys
                keys_out.append(keys[exists])
                rows_out.append(rows[exists] + start)
        if keys_out:
            keys = torch.cat(keys_out)
            rows = torch.cat(rows_out)
            keys, order = torch.sort(keys, stable=True)
            rows = rows[order]
        else:
            keys = torch.zeros(0, dtype=torch.long, device=device)
            rows = torch.zeros(0, dtype=torch.long, device=device)
        self._halo_pairs[radius] = (keys.cpu(), rows.cpu())
        return self._halo_pairs[radius]

    def plan_chunks(
        self, max_points: int, radius: int, device: torch.device
    ) -> List[Chunk]:
        """Group consecutive tiles into chunks of about ``max_points`` own voxels."""
        pair_keys, pair_rows = self.halo_pairs(radius, device)
        counts = (self.tile_start[1:] - self.tile_start[:-1]).tolist()
        chunks: List[Chunk] = []
        first, total = 0, 0
        for tile, count in enumerate(counts):
            total += count
            last_tile = tile == len(counts) - 1
            if total >= max_points or last_tile:
                start = int(self.tile_start[first])
                stop = int(self.tile_start[tile + 1])
                lo = int(torch.searchsorted(pair_keys, self.tile_keys[first], right=False))
                hi = int(torch.searchsorted(pair_keys, self.tile_keys[tile], right=True))
                halo = torch.unique(pair_rows[lo:hi])
                halo = halo[(halo < start) | (halo >= stop)]
                chunks.append(Chunk(len(chunks), start, stop, halo))
                first, total = tile + 1, 0
        return chunks


def run_chunks(
    chunks: Sequence[Chunk],
    devices: Sequence[torch.device],
    work: Callable[[Chunk, torch.device], object],
) -> List[object]:
    """Run ``work(chunk, device)`` over all chunks, one worker thread per device.

    Devices pull the next chunk when free, so a faster GPU simply takes more chunks.
    Returns the results in chunk order and re-raises the first worker exception.
    """
    results: List[object] = [None] * len(chunks)
    if len(devices) == 1:
        with torch.cuda.device(devices[0]):
            for chunk in chunks:
                results[chunk.index] = work(chunk, devices[0])
        return results

    pending: "queue.Queue[Chunk]" = queue.Queue()
    for chunk in chunks:
        pending.put(chunk)
    errors: List[BaseException] = []

    def worker(device: torch.device) -> None:
        try:
            with torch.cuda.device(device), torch.inference_mode():
                while not errors:
                    try:
                        chunk = pending.get_nowait()
                    except queue.Empty:
                        return
                    results[chunk.index] = work(chunk, device)
                torch.cuda.synchronize(device)
        except BaseException as error:  # noqa: BLE001 - re-raised on the caller
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(device,), daemon=True) for device in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def host_empty(shape: Iterable[int], dtype: torch.dtype) -> torch.Tensor:
    """Allocate pageable host memory for a level tensor.

    Not pinned: PyTorch's pinned-memory allocator caches every freed block for
    reuse, so the multi-gigabyte level tensors of successive stages would stay
    locked in RAM for the life of the process.
    """
    return torch.empty(tuple(shape), dtype=dtype)
