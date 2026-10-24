"""ComfyUI nodes wrapping the Hi3D v1.0 (SPARC3D) image-conditioned mesh refiner."""

import logging
import os

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import Types
from comfy_extras.nodes_save_3d import get_mesh_batch_item, pack_variable_mesh_batch

logger = logging.getLogger(__name__)

hi3d_models_dir = os.path.join(folder_paths.models_dir, "hi3d")
os.makedirs(hi3d_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("hi3d", hi3d_models_dir)

RESOLUTIONS = ["512", "1024", "1536"]
PASSES = {"512": 1, "1024": 2, "1536": 3}
SAME_DEVICE = "same as device"
_RELEASE_SKIP_POINTS = 12_000_000
"""Decoder residual-block bypass threshold of the released inference code."""


def _model_dirs():
    """Model directories under every registered ``hi3d`` path that hold ``model_index.json``."""
    found = []
    for root in folder_paths.get_folder_paths("hi3d"):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, name, "model_index.json")):
                found.append(name)
    return found or ["(put Hi3Dv1.0 in models/hi3d)"]


def _model_path(name):
    for root in folder_paths.get_folder_paths("hi3d"):
        path = os.path.join(root, name)
        if os.path.isfile(os.path.join(path, "model_index.json")):
            return path
    raise FileNotFoundError(f"Hi3D model directory not found under models/hi3d: {name}")


def _cuda_devices():
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cuda:0"]


def _available_device(name, fallback, label):
    """``name`` if this machine has it, else ``fallback``: workflows carry the saver's GPUs."""
    if name in _cuda_devices():
        return name
    logger.warning("Hi3D: %s %r is not available here (%s); using %s instead",
                   label, name, ", ".join(_cuda_devices()), fallback or SAME_DEVICE)
    return fallback


def _to_pil_rgba(image, mask, invert_mask):
    """ComfyUI IMAGE [H, W, C] (+ optional MASK [H, W]) to the PIL image Hi3D expects.

    Hi3D crops the subject by its alpha bounding box and composites it over white, so a
    foreground mask becomes the alpha channel. Without a mask the whole frame is used.
    """
    rgb = (image[..., :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    if mask is None:
        if image.shape[-1] == 4:
            alpha = (image[..., 3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
        return Image.fromarray(rgb, "RGB")
    m = mask.float()
    if m.shape != image.shape[:2]:
        m = torch.nn.functional.interpolate(
            m[None, None], size=tuple(image.shape[:2]), mode="bilinear", align_corners=False
        )[0, 0]
    if invert_mask:
        m = 1.0 - m
    alpha = (m.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def _fit_to_frame(vertices, reference):
    """Map ``vertices`` onto the bounding box of ``reference`` (uniform scale + translation).

    Hi3D returns geometry recentred and scaled to unit radius; downstream Trellis nodes
    (voxel texture baking, PaintMesh) sample by position, so the refined mesh must sit
    where the coarse mesh was.
    """
    lo, hi = vertices.min(0), vertices.max(0)
    ref_lo, ref_hi = reference.min(0), reference.max(0)
    scale = float((ref_hi - ref_lo).max() / max((hi - lo).max(), 1e-12))
    return (vertices - (lo + hi) / 2) * scale + (ref_lo + ref_hi) / 2


class Hi3DPipelineWrapper:
    """Holds a CPU-resident Sparc3DPipeline and the devices it should run on."""

    def __init__(self, pipeline, device, decode_device):
        self.pipeline = pipeline
        self.device = device
        self.decode_device = decode_device
        self.on_gpu = False

    def to_gpu(self):
        if not self.on_gpu:
            self.pipeline.to(self.device, decode_device=self.decode_device)
            self.on_gpu = True

    def to_cpu(self):
        if self.on_gpu:
            self.pipeline.to("cpu")
            self.on_gpu = False
            comfy.model_management.soft_empty_cache()


class Hi3DLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        devices = _cuda_devices()
        decode_devices = [SAME_DEVICE] + devices
        return {
            "required": {
                "model": (_model_dirs(), {"tooltip": "Directory under models/hi3d holding model_index.json and the transformer/VAE weights."}),
                "device": (devices, {"default": devices[0],
                                     "tooltip": "GPU for the image encoders and both 1.8B transformers (the sampling loop)."}),
                "decode_device": (decode_devices, {"default": devices[1] if len(devices) > 1 else SAME_DEVICE,
                                                   "tooltip": "GPU for VAE decoding, mesh extraction and revoxelization, "
                                                              "where memory peaks at 1024/1536. A second GPU keeps both stages within 24 GB."}),
            },
        }

    RETURN_TYPES = ("HI3D_PIPELINE",)
    RETURN_NAMES = ("hi3d_pipeline",)
    FUNCTION = "load"
    CATEGORY = "3d/Hi3D"
    DESCRIPTION = ("Loads Hi3D v1.0 on the CPU. The high-resolution transformer is read on first use; "
                   "DINOv2-large and CLIP ViT-L/14 come from the Hugging Face cache (downloaded if missing).")

    @classmethod
    def VALIDATE_INPUTS(cls, device, decode_device):
        # Taking these inputs here skips ComfyUI's check against this machine's GPU list,
        # so a workflow saved on a two-GPU machine still loads; load() falls back instead.
        return True

    def load(self, model, device, decode_device):
        device = _available_device(device, _cuda_devices()[0], "device")
        if decode_device != SAME_DEVICE:
            decode_device = _available_device(decode_device, SAME_DEVICE, "decode_device")
            if decode_device == device:
                decode_device = SAME_DEVICE
        from transformers.utils import logging as hf_logging

        from sparc3d import Sparc3DPipeline

        path = _model_path(model)
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # CLIPVisionModel reports the unused text tower at length
        try:
            try:
                pipeline = Sparc3DPipeline.from_pretrained(path, local_files_only=True)
            except OSError:
                logger.info("Hi3D: image encoders not cached, downloading from Hugging Face")
                pipeline = Sparc3DPipeline.from_pretrained(path)
        finally:
            hf_logging.set_verbosity(verbosity)
        decode = None if decode_device == SAME_DEVICE else decode_device
        return (Hi3DPipelineWrapper(pipeline, device, decode),)


class Hi3DRefineMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hi3d_pipeline": ("HI3D_PIPELINE",),
                "image": ("IMAGE", {"tooltip": "Reference image of the same object as the coarse mesh."}),
                "mesh": ("MESH", {"tooltip": "Coarse mesh (glTF Y-up, as produced by the Trellis2/Pixal3D decode)."}),
                "resolution": (RESOLUTIONS, {"default": "1024",
                                             "tooltip": "512 runs one refinement pass, 1024 two, 1536 three. "
                                                        "With streamed_decode, 1536 fits 24 GB cards; without it the "
                                                        "decode needs the whole level in GPU memory."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100, "tooltip": "Scheduler updates per pass."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "match_input_frame": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Fit the refined mesh to the coarse mesh's bounding box. Required for "
                                                             "Trellis voxel texture baking / PaintMesh; off returns Hi3D's unit-radius frame."}),
                "offload_after": ("BOOLEAN", {"default": True,
                                              "tooltip": "Move Hi3D back to CPU afterwards so other models get the VRAM."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Foreground mask; becomes the alpha Hi3D uses to frame the subject."}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "voxel_budget_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                                                 "tooltip": "Multiplies the per-pass occupied-voxel budgets (80K / 90K / 110K). "
                                                            "When a detailed object exceeds a budget the grid shrinks and fine "
                                                            "detail is lost; raising it keeps the full grid at the cost of "
                                                            "sampling time (attention grows with the square of the tokens)."}),
                "streamed_decode": ("BOOLEAN", {"default": True,
                                                "tooltip": "Decode and extract in tiles with features held in system RAM, "
                                                           "shared across the GPUs below. Removes the 24 GB decode limit, "
                                                           "so 1536 and raised budgets fit; same result as the in-memory decoder."}),
                "stream_devices": (["all GPUs", "decode device only"], {"default": "all GPUs",
                                                                         "tooltip": "GPUs that share the streamed decode tiles."}),
                "full_decoder": ("BOOLEAN", {"default": True,
                                             "tooltip": "Run every decoder residual block. The release bypasses them above "
                                                        "12M points to fit GPU memory (a quality loss at 1024/1536); "
                                                        "streamed decoding does not need that. Off reproduces the release."}),
                "decode_point_limit_m": ("INT", {"default": 0, "min": 0, "max": 1000,
                                                 "tooltip": "Refuse another pass when a decode exceeds this many million points "
                                                            "(0 = no limit with streamed decode, the release's 12M without)."}),
                "output_faces": ("INT", {"default": -1, "min": -1, "max": 500_000_000, "step": 100_000,
                                         "tooltip": "Faces of the final mesh after Hi3D's pyfqmr simplification. "
                                                    "-1 = release default (1M at 512/1024, 2M at 1536); "
                                                    "0 = no simplification, keep every extracted face "
                                                    "(tens of millions at 1536; slow to preview, unwrap and save)."}),
                "offload_idle_models": ("BOOLEAN", {"default": True,
                                                    "tooltip": "Keep only the transformer in use on the GPU; the other one and the "
                                                               "image encoders wait in system RAM. Frees about 5 GB for large budgets."}),
            },
        }

    RETURN_TYPES = ("MESH", "IMAGE")
    RETURN_NAMES = ("mesh", "conditioning_image")
    FUNCTION = "refine"
    CATEGORY = "3d/Hi3D"
    DESCRIPTION = "Refines a coarse mesh into high-resolution geometry conditioned on a reference image (geometry only)."

    def refine(self, hi3d_pipeline, image, mesh, resolution, steps, seed,
               match_input_frame, offload_after, mask=None, invert_mask=False,
               voxel_budget_scale=1.0, streamed_decode=True, stream_devices="all GPUs",
               full_decoder=True, decode_point_limit_m=0, offload_idle_models=True,
               output_faces=-1):
        import trimesh

        from sparc3d.pipelines import sparc3d_pipeline as sp
        from sparc3d.utils.image import prepare_image

        pipe = hi3d_pipeline.pipeline
        pipe.pass_budgets = tuple(int(round(b * voxel_budget_scale)) for b in sp.PASS_BUDGETS)
        pipe.offload_idle_models = bool(offload_idle_models)
        pipe.output_face_count = None if output_faces < 0 else int(output_faces)
        if streamed_decode:
            others = _cuda_devices() if stream_devices == "all GPUs" else []
            pipe.enable_streamed_decoding(others)
            default_limit = None
        else:
            pipe.disable_streamed_decoding()
            default_limit = sp.DECODE_FEATURE_CEILING
        pipe.decode_feature_ceiling = decode_point_limit_m * 1_000_000 if decode_point_limit_m else default_limit
        pipe.set_decoder_skip_points(2**62 if full_decoder else _RELEASE_SKIP_POINTS)
        logger.info("Hi3D: budgets %s, streamed=%s, full decoder=%s, point limit=%s",
                    pipe.pass_budgets, streamed_decode, full_decoder, pipe.decode_feature_ceiling)

        resolution_value = int(resolution)
        batch = int(mesh.vertices.shape[0])
        pbar = comfy.utils.ProgressBar(batch * PASSES[resolution] * steps)

        def on_progress(event):
            comfy.model_management.throw_exception_if_processing_interrupted()
            pbar.update(1)

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        pipe.progress_callback = on_progress
        out_vertices, out_faces, previews = [], [], []
        try:
            hi3d_pipeline.to_gpu()
            for i in range(batch):
                vertices, faces, *_ = get_mesh_batch_item(mesh, i)
                coarse = trimesh.Trimesh(
                    vertices=vertices.float().cpu().numpy(),
                    faces=faces.long().cpu().numpy(),
                    process=False,
                )
                image_i = image[min(i, image.shape[0] - 1)]
                mask_i = None if mask is None else mask[min(i, mask.shape[0] - 1)] if mask.ndim == 3 else mask
                pil = _to_pil_rgba(image_i, mask_i, invert_mask)
                prepared = prepare_image(pil)
                previews.append(torch.from_numpy(np.asarray(prepared.convert("RGB"), dtype=np.float32) / 255.0))

                result = pipe(pil, coarse, resolution=resolution_value, num_inference_steps=steps, seed=seed)
                refined = np.asarray(result.mesh.vertices, dtype=np.float64)
                if match_input_frame:
                    refined = _fit_to_frame(refined, coarse.vertices)
                out_vertices.append(torch.from_numpy(refined).float())
                out_faces.append(torch.from_numpy(np.asarray(result.mesh.faces)).int())
                logger.info("Hi3D: refined item %d to %d (%d faces)", i, resolution_value, len(result.mesh.faces))
        finally:
            pipe.progress_callback = None
            if offload_after:
                hi3d_pipeline.to_cpu()

        if len(out_vertices) == 1:
            out_mesh = Types.MESH(vertices=out_vertices[0][None], faces=out_faces[0][None])
        else:
            out_mesh = pack_variable_mesh_batch(out_vertices, out_faces)
        return (out_mesh, torch.stack(previews))


NODE_CLASS_MAPPINGS = {
    "Hi3DLoadModel": Hi3DLoadModel,
    "Hi3DRefineMesh": Hi3DRefineMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hi3DLoadModel": "Hi3D Load Model",
    "Hi3DRefineMesh": "Hi3D Refine Mesh",
}
