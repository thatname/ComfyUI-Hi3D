# ComfyUI-Hi3D

ComfyUI nodes for **Hi3D v1.0** (SPARC3D) image-conditioned mesh refinement: a coarse mesh
plus a reference image in, high-resolution geometry out. The inference code is vendored in
`sparc3d/` from `D:\Code\Hi3Dv1.0` (including its Windows / multi-GPU / streamed-decode
changes); its license files are kept alongside.

## Nodes

- **Hi3D Load Model** (`3d/Hi3D`) — loads a model directory from `models/hi3d/` on the CPU.
  `device` runs the image encoders and both 1.8B transformers; `decode_device` runs the
  latent-level decoder stages, mesh extraction and revoxelization.
- **Hi3D Refine Mesh** (`3d/Hi3D`) — `image` + optional foreground `mask` (used as alpha for
  framing) + coarse `MESH` → refined `MESH` and the conditioning image Hi3D actually saw.
  `match_input_frame` fits the result back onto the coarse mesh's bounding box so Trellis
  voxel texture baking and PaintMesh line up. Comfy models are unloaded before it runs, and
  `offload_after` moves Hi3D back to the CPU afterwards.

  Detail / memory controls (optional inputs):

  | input | effect |
  |---|---|
  | `voxel_budget_scale` | Multiplies the per-pass voxel budgets (80K / 90K / 110K). Detailed objects exceed them and the grid shrinks (e.g. 128 → 115), losing detail. 2.5 keeps the full 128 / 192 grids on the floating-island example. Sampling time grows roughly with the square of the tokens. |
  | `streamed_decode` | Decoder level tensors live in system RAM; every high-resolution layer runs in Morton-ordered tiles with a one-voxel halo, group-norm statistics are accumulated globally, and the tiles are shared across GPUs. Extraction is tiled the same way (three-cube halo, dual vertices welded by cube). Results match the in-memory decoder up to floating-point summation order (verified: same point/face counts, differences at the stock decoder's own run-to-run level). Device memory stays under ~10 GB. |
  | `stream_devices` | `all GPUs` shares decode/extraction tiles across every GPU. |
  | `full_decoder` | Runs the decoder residual blocks the release skips above 12M points (it skips them only to fit GPU memory). Off reproduces the release exactly. |
  | `decode_point_limit_m` | Refuse another pass above this many million decoded points; 0 = no limit when streamed (release: 12M). |
  | `offload_idle_models` | Only the transformer in use stays on the GPU; the other and the image encoders wait in RAM. |

## Requirements

- Model: `models/hi3d/Hi3Dv1.0/` with `model_index.json`, `scheduler/`, `transformer_512/`,
  `transformer_1024/`, `vae/` (here a junction to `D:\Code\Hi3Dv1.0\ckpt\Hi3Dv1.0`).
  `facebook/dinov2-large` and `openai/clip-vit-large-patch14` load from the Hugging Face
  cache, downloading on first use if missing.
- Python packages: `spconv-cu126` + `cumm-cu126` (Windows: needs a CUDA 12.x toolkit
  installed, found via `CUDA_PATH_V12_*`), `flash-attn`, Triton (`triton-windows`),
  `transformers`, `safetensors`, `trimesh`, `pyfqmr`, `scipy`.
- RAM: 1536 at `voxel_budget_scale` 2.5 on a detailed object peaks around 55 GB in the
  final simplification (an 86M-face mesh); lower the budget scale if RAM is short.

## Workflow

`user/default/workflows/3d_pixal3d_trellis2_hi3d_refine_image_to_model.json` — the
Trellis2 / Pixal3D image-to-model workflow with Hi3D inserted between the shape decode and
mesh post-processing, set to 1536 with budget scale 2.5. `RemeshMesh` is bypassed (a 768
remesh would erase the refinement) and `DecimateMesh` is set to 2M faces: without the remesh
its GPU QEM overshoots on Hi3D's already-simplified mesh (2M -> 342K for a 700K target), so
the saved asset keeps Hi3D's own 2M faces (~210 MB GLB). Lower it for a lighter asset.

Measured on the floating-island example, RTX 4090 (transformers) + RTX 3090, 128 GB RAM:

| setting | pass grids | decoded points (last pass) | Hi3D time |
|---|---|---|---|
| 1024, release settings | 64 / 115 (shrunk) | 14.5M, two decoder blocks skipped | ~7 min (195 s of it hole filling, now ~20 s) |
| 1536, scale 2.5, streamed | 64 / 128 / 192 | 48M, all blocks | ~16 min (pass-3 sampling 8 min) |

The transformer stays on one GPU: splitting classifier-free guidance across the 4090 and
3090 does not help, because a 3090 forward takes 2.2x as long as a 4090 forward (11.5 s vs
5.2 s at 250K tokens) and the guided pair would wait on it.
