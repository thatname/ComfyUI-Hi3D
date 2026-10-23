"""Batch-1 in-memory SPARC3D mesh refinement pipeline."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

from .output import Sparc3DPipelineOutput
from .progress import Sparc3DProgressEvent


logger = logging.getLogger(__name__)

ProgressCallback = Callable[[Sparc3DProgressEvent], None]
"""Callback invoked after each completed scheduler update."""

STRUCTURE_RESOLUTIONS = (64, 128, 192)
"""Sparse structure resolution of each rung of the refinement ladder."""

OUTPUT_RESOLUTIONS = (512, 1024, 1536)
"""Mesh resolution produced by each rung, parallel to :data:`STRUCTURE_RESOLUTIONS`."""

PASS_BUDGETS = (80_000, 90_000, 110_000)
"""Occupied-voxel budget adaptive voxelization targets at each rung."""

TARGET_FACE_COUNTS = (1_000_000, 1_000_000, 2_000_000)
"""Face count the final mesh is simplified to, per :data:`OUTPUT_RESOLUTIONS` entry."""

DECODE_FEATURE_CEILING = 12_000_000
"""Decoded feature rows above which a further pass would exhaust device memory."""

COMPONENT_FACE_RATIO = 0.005
"""Connected components holding a smaller share of faces are dropped as debris."""


class Sparc3DPipeline:
    """Public sparse refinement pipeline with lazy high-resolution residency."""

    def __init__(
        self,
        *,
        transformer_512: Any,
        vae: Any,
        conditioner: Any,
        scheduler: Any,
        transformer_1024: Any | None = None,
        transformer_1024_loader: Callable[[], Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Assemble already-loaded pipeline components.

        Args:
            transformer_512: Transformer used for structure resolution 64.
            vae: Sparse VAE used to decode every completed pass.
            conditioner: Shared image conditioner.
            scheduler: Flow-matching scheduler; second-order schedulers declare
                ``order = 2`` and have their corrector slope evaluated here.
            transformer_1024: Optional materialized high-resolution transformer.
            transformer_1024_loader: Optional one-shot lazy loader for that transformer.
            progress_callback: Optional callback stored on the pipeline object.

        Raises:
            ValueError: Both or neither high-resolution transformer sources are given.
        """
        if (transformer_1024 is None) == (transformer_1024_loader is None):
            raise ValueError(
                "provide exactly one of transformer_1024 or transformer_1024_loader"
            )
        if transformer_1024_loader is not None and not callable(
            transformer_1024_loader
        ):
            raise TypeError("transformer_1024_loader must be callable")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable")
        self.transformer_512 = transformer_512
        self.vae = vae
        self.conditioner = conditioner
        self.scheduler = scheduler
        self._transformer_1024 = transformer_1024
        self._transformer_1024_loader = transformer_1024_loader
        self.progress_callback = progress_callback
        self._device = torch.device("cpu")
        self._decode_device = torch.device("cpu")
        self._sequential_cpu_offload = False
        self.pass_budgets: tuple[int, ...] = PASS_BUDGETS
        """Occupied-voxel budget per rung; raising it trades sampling time for detail."""
        self.decode_feature_ceiling: int | None = DECODE_FEATURE_CEILING
        """Decoded rows above which another pass is refused; ``None`` disables it."""
        self.offload_idle_models = False
        """Keep only the components a stage is using on the GPU."""
        self._streaming_devices: list[torch.device] | None = None
        self._chunk_bytes = 0
        self._streamer: Any = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        dino_model_id: str | Path | None = None,
        clip_model_id: str | Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> "Sparc3DPipeline":
        """Load a safe local or Hugging Face pipeline artifact.

        Args:
            pretrained_model_name_or_path: Local model directory or Hub repository ID.
            revision: Optional model-repository revision.
            cache_dir: Optional Hugging Face cache directory, meaning the directory
                that holds the ``models--*`` trees rather than ``$HF_HOME`` itself.
            local_files_only: Whether Hub loading must use cached files only.
            dino_model_id: Optional DINOv2 repository ID or local directory. The
                conditioning encoders live outside the pipeline artifact, so this
                is how an offline host points at its own copy.
            clip_model_id: Optional CLIP repository ID or local directory.
            progress_callback: Optional callback stored on the pipeline object.

        Returns:
            CPU-resident pipeline with the 1024 transformer still lazy.
        """
        from . import loader

        root = loader.resolve_model_directory(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            defer_transformer_1024=True,
        )
        transformer_512 = loader.load_transformer(root, "transformer_512")
        vae = loader.load_vae(root)
        conditioner = loader.load_conditioner(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            dino_model_id=dino_model_id,
            clip_model_id=clip_model_id,
        )
        scheduler = loader.load_scheduler(root)
        return cls(
            transformer_512=transformer_512,
            transformer_1024_loader=lambda: loader.load_deferred_transformer_1024(
                root,
                pretrained_model_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            ),
            vae=vae,
            conditioner=conditioner,
            scheduler=scheduler,
            progress_callback=progress_callback,
        )

    @property
    def transformer_1024(self) -> Any:
        """Return the high-resolution transformer, materializing it once."""
        if self._transformer_1024 is None:
            if self._transformer_1024_loader is None:
                raise RuntimeError("the transformer_1024 loader has already been consumed")
            logger.info("Loading transformer_1024 on CPU")
            model = self._transformer_1024_loader()
            self._transformer_1024_loader = None
            if self._device.type != "cpu":
                model.to(self._device)
            self._transformer_1024 = model
        return self._transformer_1024

    @property
    def device(self) -> torch.device:
        """Return the execution device selected with :meth:`to`."""
        return self._device

    @property
    def decode_device(self) -> torch.device:
        """Return the device running VAE decoding, extraction, and revoxelization."""
        return self._decode_device

    def to(
        self,
        device: str | torch.device,
        decode_device: str | torch.device | None = None,
    ) -> "Sparc3DPipeline":
        """Move materialized components to ``device`` without changing dtype.

        Args:
            device: Execution device accepted by ``torch.device``, used by the
                conditioner and both transformers.
            decode_device: Optional second device for the VAE decoder, mesh
                extraction, and revoxelization, whose memory peaks at high
                resolution. Defaults to ``device``.

        Returns:
            This pipeline.

        Raises:
            RuntimeError: Sequential CPU offload hooks are active.
        """
        if self._sequential_cpu_offload:
            raise RuntimeError("cannot call to() after enabling sequential CPU offload")
        target = torch.device(device)
        decode_target = target if decode_device is None else torch.device(decode_device)
        self.transformer_512.to(target)
        self.vae.to(decode_target)
        self.conditioner.to(target)
        if self._transformer_1024 is not None:
            self._transformer_1024.to(target)
        self._device = target
        self._decode_device = decode_target
        self._release_streamer()
        return self

    def enable_streamed_decoding(
        self,
        devices: list[str | torch.device] | None = None,
        chunk_bytes: int | None = None,
    ) -> "Sparc3DPipeline":
        """Decode and extract out of core, in tiles shared across ``devices``.

        Level tensors are kept in host memory and each tile visits a GPU only while
        it is computed, so decoding is no longer bounded by one device's memory and
        the feature ceiling can be raised. Results match the in-memory decoder up to
        floating-point summation order.

        Args:
            devices: Additional CUDA devices sharing the tiles. The decode device is
                always included and runs the latent-level attention stack.
            chunk_bytes: Approximate device working set of one tile chunk.

        Returns:
            This pipeline.
        """
        from sparc3d.models.autoencoders.streaming import DEFAULT_CHUNK_BYTES

        self._streaming_devices = [torch.device(device) for device in (devices or [])]
        self._chunk_bytes = int(chunk_bytes or DEFAULT_CHUNK_BYTES)
        self._release_streamer()
        return self

    def disable_streamed_decoding(self) -> "Sparc3DPipeline":
        """Return to decoding each pass in device memory on the decode device."""
        self._streaming_devices = None
        self._release_streamer()
        return self

    def set_decoder_skip_points(self, skip_points: int) -> "Sparc3DPipeline":
        """Set the point count above which decoder residual blocks are bypassed.

        The release bypasses them above 12M points to fit device memory; streamed
        decoding has no such limit, so a larger value runs the full network.
        """
        from sparc3d.models.autoencoders.blocks import SparseResnetBlock

        for module in self.vae.modules():
            if isinstance(module, SparseResnetBlock):
                module.skip_points = int(skip_points)
        self._release_streamer()
        return self

    def _release_streamer(self) -> None:
        if self._streamer is not None:
            self._streamer.release()
            self._streamer = None

    def _stream_devices(self) -> list[torch.device]:
        devices = [self._decode_device]
        for device in self._streaming_devices or []:
            if device not in devices:
                devices.append(device)
        return devices

    def enable_sequential_cpu_offload(
        self, device: str | torch.device = "cuda"
    ) -> "Sparc3DPipeline":
        """Install Accelerate CPU-offload hooks on every inference component.

        The lazy high-resolution transformer is materialized on the CPU before hooks
        are installed so no unhooked model can enter a later pass.

        Args:
            device: Non-CPU execution device used while each module runs.

        Returns:
            This pipeline with real Accelerate hooks installed.

        Raises:
            ValueError: ``device`` is a CPU device.
            RuntimeError: Offload is already enabled or components were moved first.
            ImportError: Accelerate is unavailable.
        """
        if self._sequential_cpu_offload:
            raise RuntimeError("sequential CPU offload is already enabled")
        execution_device = torch.device(device)
        if execution_device.type == "cpu":
            raise ValueError("sequential CPU offload requires a non-CPU device")
        if self._device.type != "cpu":
            raise RuntimeError("enable offload while all components are still on CPU")
        try:
            from accelerate import cpu_offload
        except ImportError as exc:
            raise ImportError(
                "Sequential CPU offload requires the accelerate package."
            ) from exc

        for model in (
            self.conditioner,
            self.transformer_512,
            self.transformer_1024,
            self.vae,
        ):
            cpu_offload(
                model,
                execution_device=execution_device,
                offload_buffers=True,
            )
        self._device = execution_device
        self._decode_device = execution_device
        self._sequential_cpu_offload = True
        return self

    @torch.inference_mode()
    def __call__(
        self,
        image: Any,
        coarse_mesh: Any,
        resolution: int = 1024,
        num_inference_steps: int = 25,
        generator: torch.Generator | None = None,
        seed: int | None = None,
    ) -> Sparc3DPipelineOutput:
        """Refine an in-memory coarse mesh through the requested resolution.

        Args:
            image: Batch-1 PIL image.
            coarse_mesh: In-memory ``trimesh.Trimesh`` or transformed scene.
            resolution: Requested output resolution.
            num_inference_steps: Scheduler updates run in each pass.
            generator: Optional local CPU random generator.
            seed: Optional seed used to create a local CPU generator.

        Returns:
            Geometry-only mesh at the requested resolution.
        """
        local_generator = self._resolve_generator(generator, seed)
        self._validate_steps(num_inference_steps)
        self._validate_target_resolution(resolution)

        from sparc3d.sparcubes.postprocess import validate_mesh
        from sparc3d.utils.image import prepare_image

        prepared = prepare_image(image)
        mesh = validate_mesh(coarse_mesh)
        condition = self._condition(prepared)
        coords = self._adaptive_voxelize(mesh, 64, self.pass_budgets[0])
        return self._run_refinement(
            condition,
            coords,
            structure_resolution=64,
            resolution=resolution,
            num_inference_steps=num_inference_steps,
            generator=local_generator,
        )

    @torch.inference_mode()
    def refine_voxels(
        self,
        image: Any,
        voxel_coords: Any,
        structure_resolution: int,
        resolution: int,
        num_inference_steps: int = 25,
        generator: torch.Generator | None = None,
        seed: int | None = None,
    ) -> Sparc3DPipelineOutput:
        """Refine validated batchless voxel coordinates in memory.

        Args:
            image: Batch-1 PIL image.
            voxel_coords: Unique integer coordinates shaped ``[N, 3]``.
            structure_resolution: Grid resolution represented by ``voxel_coords``.
            resolution: Requested output resolution.
            num_inference_steps: Scheduler updates run in each pass.
            generator: Optional local CPU random generator.
            seed: Optional seed used to create a local CPU generator.

        Returns:
            Geometry-only mesh at the requested resolution.
        """
        local_generator = self._resolve_generator(generator, seed)
        self._validate_steps(num_inference_steps)
        coords = self._validate_voxel_coords(voxel_coords, structure_resolution)
        self._validate_start_target(structure_resolution, resolution)

        from sparc3d.utils.image import prepare_image

        prepared = prepare_image(image)
        condition = self._condition(prepared)
        return self._run_refinement(
            condition,
            coords,
            structure_resolution=structure_resolution,
            resolution=resolution,
            num_inference_steps=num_inference_steps,
            generator=local_generator,
        )

    def _run_refinement(
        self,
        condition: torch.Tensor,
        voxel_coords: torch.Tensor,
        *,
        structure_resolution: int,
        resolution: int,
        num_inference_steps: int,
        generator: torch.Generator,
    ) -> Sparc3DPipelineOutput:
        start_index = STRUCTURE_RESOLUTIONS.index(structure_resolution)
        target_index = OUTPUT_RESOLUTIONS.index(resolution)
        pass_indices = list(range(start_index, target_index + 1))
        mesh = None
        coords = voxel_coords

        for call_pass_index, ladder_index in enumerate(pass_indices):
            structure = STRUCTURE_RESOLUTIONS[ladder_index]
            transformer = (
                self.transformer_512 if structure == 64 else self.transformer_1024
            )
            if self.offload_idle_models:
                self._keep_on_device(transformer)
            with self._on(self._device):
                latents = self._sample_latents(
                    transformer,
                    condition,
                    coords,
                    structure,
                    num_inference_steps,
                    generator,
                    pass_index=call_pass_index,
                    num_passes=len(pass_indices),
                )
            with self._on(self._decode_device):
                decoded = self._decode_latents(latents, structure)
                has_another_pass = ladder_index < target_index
                decoded_features = self._decoded_feature_count(decoded)
                ceiling = self.decode_feature_ceiling
                if has_another_pass and ceiling is not None and decoded_features > ceiling:
                    raise RuntimeError(
                        "decoded feature ceiling exceeded before another pass: "
                        f"{decoded_features} > {ceiling}"
                    )
                mesh = self._extract_mesh(decoded)
                del decoded

            if has_another_pass:
                next_index = ladder_index + 1
                coords = self._adaptive_voxelize(
                    mesh,
                    STRUCTURE_RESOLUTIONS[next_index],
                    self.pass_budgets[next_index],
                )

        if mesh is None:
            raise RuntimeError("refinement ladder completed without producing a mesh")
        mesh = self._postprocess_mesh(
            mesh, target_face_count=TARGET_FACE_COUNTS[target_index]
        )
        return Sparc3DPipelineOutput(mesh=mesh, resolution=resolution)

    def _sample_latents(
        self,
        transformer: Any,
        condition: torch.Tensor,
        voxel_coords: torch.Tensor,
        structure_resolution: int,
        num_inference_steps: int,
        generator: torch.Generator,
        *,
        pass_index: int,
        num_passes: int,
    ) -> Any:
        from sparc3d.nn.sparse.basic import SparseTensor

        coords = voxel_coords.to(device=self._device, dtype=torch.int32)
        batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=self._device)
        batched_coords = torch.cat((batch, coords), dim=1)
        noise = torch.randn(
            (coords.shape[0], 32),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(self._device)
        sample = SparseTensor(feats=noise, coords=batched_coords)
        negative_condition = torch.zeros_like(condition)
        timesteps = self.scheduler.set_timesteps(
            num_inference_steps,
            num_tokens=coords.shape[0],
            device=self._device,
        )

        second_order = getattr(self.scheduler, "order", 1) == 2

        for step_index, sigma_hat in enumerate(timesteps):
            velocity = self._guided_velocity(
                transformer, sample, sigma_hat, condition, negative_condition
            )
            if second_order:
                # Both slopes are guided at their own time, so an update that
                # crosses the guidance threshold pays for three forward passes.
                predicted, next_sigma = self.scheduler.predict(
                    velocity, sigma_hat, sample.feats
                )
                second_velocity = self._guided_velocity(
                    transformer,
                    sample.replace(predicted),
                    next_sigma,
                    condition,
                    negative_condition,
                )
                updated = self.scheduler.step(
                    velocity, sigma_hat, sample.feats, second_output=second_velocity
                )
            else:
                updated = self.scheduler.step(velocity, sigma_hat, sample.feats)
            sample = sample.replace(updated)
            if self.progress_callback is not None:
                self.progress_callback(
                    Sparc3DProgressEvent(
                        pass_index=pass_index,
                        num_passes=num_passes,
                        structure_resolution=structure_resolution,
                        step=step_index + 1,
                        num_inference_steps=num_inference_steps,
                    )
                )

        mean, std = self._latent_stats(transformer, sample)
        return sample.replace(sample.feats * std + mean)

    @staticmethod
    def _guided_velocity(
        transformer: Any,
        sample: Any,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        negative_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one flow velocity, guided only from time 0.7 upwards."""
        from sparc3d.schedulers.schedule import apply_effective_guidance

        model_timestep = (1000.0 * timestep).reshape(1)
        positive = transformer(sample, model_timestep, condition)
        if float(timestep) < 0.7:
            return positive.feats
        negative = transformer(sample, model_timestep, negative_condition)
        return apply_effective_guidance(positive.feats, negative.feats, timestep)

    def _condition(self, prepared: Any) -> torch.Tensor:
        if self.offload_idle_models and self._device.type == "cuda":
            self.conditioner.to(self._device)
        with self._on(self._device):
            return self.conditioner(prepared).to(self._device)

    def _keep_on_device(self, transformer: Any) -> None:
        """Move the transformer about to sample onto the device and the rest off it."""
        if self._device.type != "cuda":
            return
        idle = [self.conditioner, self.transformer_512]
        if self._transformer_1024 is not None:
            idle.append(self._transformer_1024)
        for model in idle:
            if model is not transformer:
                model.to("cpu")
        transformer.to(self._device)
        torch.cuda.empty_cache()

    def _decode_latents(self, latents: Any, structure_resolution: int) -> Any:
        grid_coord = latents.coords[:, 1:].int().to(self._decode_device)
        mapping = {
            "grid_coord": grid_coord,
            "feat": latents.feats.float().to(self._decode_device),
            "sdf": None,
            "deform": None,
            "resolution": structure_resolution,
            "offset": torch.tensor(
                [grid_coord.shape[0]], dtype=torch.int64, device=grid_coord.device
            ),
        }
        if self._streaming_devices is not None and self._decode_device.type == "cuda":
            from sparc3d.models.autoencoders.streaming import StreamedDecoder

            if self._streamer is None:
                self._streamer = StreamedDecoder(
                    self.vae.decoder, self._stream_devices(), self._chunk_bytes
                )
            return self._streamer.decode(mapping)
        with torch.autocast(
            device_type=self._decode_device.type,
            enabled=self._decode_device.type == "cuda",
        ):
            return self.vae.decode(mapping)

    @staticmethod
    def _decoded_feature_count(decoded: Any) -> int:
        features = decoded["feat"] if isinstance(decoded, dict) else decoded.feat
        return int(features.shape[0])

    def _extract_mesh(self, decoded: Any) -> Any:
        import trimesh

        from sparc3d.models.autoencoders.streaming import StreamedDecoding
        from sparc3d.sparcubes.extract import point_to_mesh
        from sparc3d.sparcubes.fill_hole import fill_mesh_holes

        if isinstance(decoded, StreamedDecoding):
            from sparc3d.sparcubes.tiled_extract import tiled_point_to_mesh

            mesh = tiled_point_to_mesh(decoded, self._stream_devices())
            mesh = fill_mesh_holes(mesh, device=self._decode_device)
        else:
            mesh = fill_mesh_holes(point_to_mesh(decoded))
        if not mesh.success:
            raise RuntimeError("mesh extraction produced an empty isosurface")
        return trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            process=False,
        )

    def _adaptive_voxelize(
        self, mesh: Any, structure_resolution: int, token_budget: int
    ) -> torch.Tensor:
        from sparc3d.ops import adaptive_vox

        with self._on(self._decode_device):
            return adaptive_vox(
                mesh,
                resolution=structure_resolution,
                target_num=token_budget,
                device=self._decode_device,
            )

    @staticmethod
    def _on(device: torch.device) -> contextlib.AbstractContextManager:
        """Make ``device`` current, as spconv and Triton launch on the current device."""
        if device.type == "cuda":
            return torch.cuda.device(device)
        return contextlib.nullcontext()

    def _postprocess_mesh(self, mesh: Any, *, target_face_count: int) -> Any:
        from sparc3d.sparcubes.postprocess import postprocess_mesh

        return postprocess_mesh(
            mesh,
            target_face_count=target_face_count,
            component_face_ratio=COMPONENT_FACE_RATIO,
            device=self._decode_device if self._decode_device.type == "cuda" else None,
        )

    @staticmethod
    def _latent_stats(
        transformer: Any, sample: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = transformer.config
        mean = torch.as_tensor(
            config.latent_mean, dtype=sample.dtype, device=sample.device
        ).reshape(1, -1)
        std = torch.as_tensor(
            config.latent_std, dtype=sample.dtype, device=sample.device
        ).reshape(1, -1)
        if mean.shape[1] != sample.feats.shape[1]:
            raise ValueError("transformer latent statistics do not match latent channels")
        return mean, std

    @staticmethod
    def _resolve_generator(
        generator: torch.Generator | None, seed: int | None
    ) -> torch.Generator:
        if generator is not None and seed is not None:
            raise ValueError("generator and seed are mutually exclusive")
        if generator is not None:
            if not isinstance(generator, torch.Generator):
                raise TypeError("generator must be a torch.Generator")
            if torch.device(generator.device).type != "cpu":
                raise ValueError("generator must be a CPU generator")
            return generator
        local_generator = torch.Generator(device="cpu")
        if seed is None:
            local_generator.seed()
        else:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise ValueError(f"seed must be an integer, got {seed!r}")
            local_generator.manual_seed(int(seed))
        return local_generator

    @staticmethod
    def _validate_voxel_coords(
        voxel_coords: Any, structure_resolution: int
    ) -> torch.Tensor:
        if (
            isinstance(structure_resolution, bool)
            or not isinstance(structure_resolution, Integral)
            or int(structure_resolution) not in STRUCTURE_RESOLUTIONS
        ):
            raise ValueError(
                "structure_resolution must be one of "
                + ", ".join(str(value) for value in STRUCTURE_RESOLUTIONS)
            )
        if not isinstance(voxel_coords, torch.Tensor):
            raise TypeError("voxel_coords must be a torch.Tensor")
        if voxel_coords.ndim != 2 or voxel_coords.shape[1] != 3:
            raise ValueError(
                f"voxel_coords must have shape [N, 3], got {tuple(voxel_coords.shape)}"
            )
        if voxel_coords.shape[0] == 0:
            raise ValueError("voxel_coords must not be empty")
        if voxel_coords.dtype == torch.bool or voxel_coords.is_floating_point():
            raise ValueError("voxel_coords must use an integer dtype")
        coords = voxel_coords.to(dtype=torch.int64)
        if torch.any(coords < 0) or torch.any(coords >= int(structure_resolution)):
            raise ValueError(
                f"voxel_coords must be in [0, {int(structure_resolution)})"
            )
        if torch.unique(coords, dim=0).shape[0] != coords.shape[0]:
            raise ValueError("voxel_coords must be unique")
        return coords.to(dtype=torch.int32)

    @staticmethod
    def _validate_steps(num_inference_steps: int) -> None:
        if (
            isinstance(num_inference_steps, bool)
            or not isinstance(num_inference_steps, Integral)
            or int(num_inference_steps) <= 0
        ):
            raise ValueError("num_inference_steps must be a positive integer")

    @staticmethod
    def _validate_target_resolution(resolution: int) -> None:
        if (
            isinstance(resolution, bool)
            or not isinstance(resolution, Integral)
            or int(resolution) not in OUTPUT_RESOLUTIONS
        ):
            raise ValueError(
                "resolution must be one of "
                + ", ".join(str(value) for value in OUTPUT_RESOLUTIONS)
            )

    @classmethod
    def _validate_start_target(
        cls, structure_resolution: int, resolution: int
    ) -> None:
        cls._validate_target_resolution(resolution)
        # A pass can only start at or below the rung its structure resolution names.
        start_index = STRUCTURE_RESOLUTIONS.index(int(structure_resolution))
        if resolution not in OUTPUT_RESOLUTIONS[start_index:]:
            raise ValueError(
                f"structure resolution {structure_resolution} cannot target {resolution}"
            )
