# ComfyUI-Hi3D

ComfyUI nodes for **Hi3D v1.0** (SPARC3D) image-conditioned mesh refinement: a coarse mesh
plus a reference image in, high-resolution geometry out. The inference code is vendored in
`sparc3d/` from `D:\Code\Hi3Dv1.0` (including its Windows / multi-GPU / streamed-decode
changes); its license files are kept alongside.

## Nodes

- **Hi3D Load Model** (`3d/Hi3D`) — loads a model directory from `models/hi3d/` on the CPU.
  `device` runs the image encoders and both 1.8B transformers; `decode_device` runs the
  latent-level decoder stages, mesh extraction and revoxelization. A device the machine
  does not have (a workflow saved on a two-GPU machine, opened on one GPU) falls back to
  the first GPU / "same as device" with a warning instead of failing validation.
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
  | `output_faces` | Face count of Hi3D's final pyfqmr simplification. -1 = release default (1M at 512/1024, 2M at 1536); 0 = no simplification, the full extracted mesh (tens of millions of faces at 1536) is returned. |
  | `offload_idle_models` | Only the transformer in use stays on the GPU; the other and the image encoders wait in RAM. |

- **Trellis2 Encode Mesh (Shape Latent)** (`3d/Hi3D`) — `MESH` → native Trellis2 `shape_latent`
  for **Trellis2 Texture Stage** and `shape_subdivides` for **Trellis2 VAE Decode Texture**.
  ComfyUI ships the Trellis2 shape decoder only, so without this the texture stage can only
  run on Trellis2's own shape latent, and its colours are generated for that shape rather
  than the refined one. The node converts the mesh to TRELLIS.2's flexible dual grid
  (`o_voxel`, installed with ComfyUI-Trellis2) at 1024 or 1536, and runs the official
  `shape_enc_next_dc_f16c32_fp16` encoder (TRELLIS.2-4B, downloaded by ComfyUI-Trellis2 to
  `models/microsoft/TRELLIS.2-4B/ckpts/`) rebuilt on ComfyUI's native sparse modules. The
  subdivisions are read off the voxels, as official TRELLIS.2 texturing does, so no extra
  shape decode is needed. `reference_latent` (the shape latent the mesh came from) supplies
  the model frame: Pixal3D is Y-up, Trellis2 Z-up.

  Checked against the ComfyUI-Trellis2 encoder on the same dual grid: identical coords,
  latent difference 6e-4 relative (fp16). Decoding the latent with the native shape decoder
  reproduces the mesh to ~0.6 voxel on average, and the subdivisions agree with the decoder's
  own predictions to >99%. 1536 on a 2M-face mesh: ~23 s, 7.6 GB peak.

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

## Workflows

Both in `example_workflows/` (copied to `user/default/workflows/`), set to 1536 with budget
scale 2.5 and `output_faces` -1 (Hi3D's own 2M simplification). `decode_device` is saved as
"same as device" so they run on a single GPU; with more GPUs, streamed decoding still shares
its tiles across all of them (`stream_devices`):

- `3d_pixal3d_trellis2_hi3d_refine_image_to_model.json`: Trellis2 runs its full cascade
  (512 shape, then its 1536 upsample stage) and Hi3D refines the 1536 decode.
- `3d_pixal3d_trellis2_lowres_hi3d_refine_image_to_model.json`: Trellis2's upsample stage
  and its sampler are removed. The 512 shape latent is decoded directly and Hi3D does the
  whole climb to 1536.

In both, **Trellis2 Encode Mesh** re-encodes the Hi3D mesh at 1536 (a 96³ latent grid, as
after the upsample stage); its latent drives the texture stage and its subdivisions the
texture decode.

In both, Hi3D's pyfqmr simplifies its extracted mesh (~95M faces on the floating island) to
2M, which is unwrapped, textured and saved; `DecimateMesh` (2M) then has nothing left to do.
Raising `output_faces` (e.g. 10M) gives the normal / AO bakes a denser source, but the 10M -> 2M
reduction then falls to ComfyUI's GPU `DecimateMesh`, which undershoots its target (~1.5M) and
collapses flat surfaces into large facets. `output_faces` 0 (no simplification) runs
`DecimateMesh` out of GPU memory on a 24 GB card with ~100M faces. `RemeshMesh` is bypassed
(a 768 remesh would erase the refinement).

Measured on the floating-island example, RTX 4090 (transformers) + RTX 3090, 128 GB RAM:

| setting | pass grids | decoded points (last pass) | Hi3D time |
|---|---|---|---|
| 1024, release settings | 64 / 115 (shrunk) | 14.5M, two decoder blocks skipped | ~7 min (195 s of it hole filling, now ~20 s) |
| 1536, scale 2.5, streamed | 64 / 128 / 192 | 48M, all blocks | ~16 min (pass-3 sampling 8 min) |

The transformer stays on one GPU: splitting classifier-free guidance across the 4090 and
3090 does not help, because a 3090 forward takes 2.2x as long as a 4090 forward (11.5 s vs
5.2 s at 250K tokens) and the guided pair would wait on it.
