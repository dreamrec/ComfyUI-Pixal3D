# Pixal3D Nodes — full parameter reference

Three nodes register under category **`Pixal3D`**. In practice you only need
the middle one (`Image to Mesh`); the others are for explicit warm-up and
VRAM cleanup.

<p align="center">
  <img src="images/nodes_preview.png" alt="Pixal3D nodes preview" width="100%">
</p>

---

## `Pixal3D: Image to Mesh`

Single image → textured PBR trimesh. The pipeline runs:

1. Optional alpha attach (from your mask)
2. BiRefNet background removal (if no mask supplied)
3. Smart crop + composite onto `background_color`
4. MoGe-2 camera intrinsics estimation
5. 4× DINOv3 projection-conditioning
6. SS / shape-LR / shape-HR / tex-LR / tex-HR sampling cascade
7. Mesh extraction + PBR texture bake + decimation + remeshing + GLB export

### Required inputs

| Name | Type | Default | Notes |
|---|---|---|---|
| `image` | IMAGE | — | First batch entry is used. |
| `pipeline_type` | enum | `1024_cascade` | `1024_cascade` (recommended) or `1536_cascade` (heavier, marginally higher fidelity). |
| `seed` | INT | `42` | Same seed + same params → identical output. |

### Socket-only optional inputs

| Name | Type | Notes |
|---|---|---|
| `mask` | MASK | When connected, BiRefNet is **skipped**. Your mask attaches as alpha, then smart-crop + composite. Recommended for complex compositions; pair with RMBG-2.0 / SAM. |
| `pipeline_ready` | PIXAL3D_PIPELINE_READY | Output of `Pixal3D: Load Pipeline`. Makes warm-up timing visible in its own node. |

### Pipeline + VRAM widgets

| Name | Type | Default | Range | What it does |
|---|---|---|---|---|
| `low_vram` | BOOL | `false` | — | Offload sub-models to CPU between stages. Slow (~2× wall time). Only enable on cards < 16 GB. |
| `max_num_tokens` | INT | `49152` | 16384-131072, step 1024 | HR token budget. Lower = less VRAM, smaller mesh detail. `49152` ≈ 9 GB activations. |
| `mesh_scale` | FLOAT | `1.0` | 0.1-5.0 | Object physical scale (used for back-projection). Leave at 1.0 unless you know what you're doing. |
| `image_resolution` | INT | `512` | 256-2048, step 64 | Internal resolution for MoGe distance calculation. |

### Camera widgets

| Name | Type | Default | Range | What it does |
|---|---|---|---|---|
| `camera_angle_x_override` | FLOAT | `0.0` | 0.0-3.14 rad | `0` = auto via MoGe-2. Set both override widgets > 0 to bypass MoGe. |
| `distance_override` | FLOAT | `0.0` | 0.0-50.0 | `0` = auto. Both override widgets must be > 0 to take effect. |

### Sparse-structure sampling (Stage 1)

Builds 3D voxel scaffolding from the image.

| Name | Type | Default | Range | What it does |
|---|---|---|---|---|
| `ss_guidance_strength` | FLOAT | `7.5` | 1.0-20.0 | CFG strength. Higher = stronger image adherence. |
| `ss_sampling_steps` | INT | `12` | 1-50 | 12 is the sweet spot; 4-8 quick previews; 16-24 high fidelity at ~2× cost. |
| `ss_guidance_rescale` | FLOAT | `0.7` | 0.0-1.0 | CFG rescale (anti-saturation). |
| `ss_rescale_t` | FLOAT | `5.0` | 0.0-10.0 | Time-rescaling exponent. |

### Shape SLat sampling (Stage 2)

Per-voxel geometry latent from SS scaffolding.

| Name | Type | Default | Range |
|---|---|---|---|
| `shape_guidance_strength` | FLOAT | `7.5` | 1.0-20.0 |
| `shape_sampling_steps` | INT | `12` | 1-50 |
| `shape_guidance_rescale` | FLOAT | `0.5` | 0.0-1.0 |
| `shape_rescale_t` | FLOAT | `3.0` | 0.0-10.0 |

### Texture SLat sampling (Stage 3)

PBR texture latent (base_color + metallic + roughness + alpha).

| Name | Type | Default | Range | What it does |
|---|---|---|---|---|
| `tex_guidance_strength` | FLOAT | `1.0` | 1.0-20.0 | Default 1.0 (no CFG) — texture is well-conditioned by shape + image. Raising can sharpen but also saturate. |
| `tex_sampling_steps` | INT | `12` | 1-50 | |
| `tex_guidance_rescale` | FLOAT | `0.0` | 0.0-1.0 | |
| `tex_rescale_t` | FLOAT | `3.0` | 0.0-10.0 | |

### Mesh extraction + baking

| Name | Type | Default | Range | What it does |
|---|---|---|---|---|
| `decimation_target` | INT | `200000` | 5000-1000000, step 5000 | Target face count post-decimation. Higher = denser mesh, slower viewer load. 200k = good viewer default; drop to 50k–100k for game engines. |
| `texture_size` | INT | `2048` | 256-4096, step 256 | PBR atlas resolution. 4096 helps when UV seams are visible. |
| `remesh` | BOOL | `true` | — | o_voxel post-remesh pass (triangle quality). Disable only if you can isolate an artifact to it. |
| `background_color` | enum | `gray` | `gray` / `black` / `white` | **Keep on `gray`.** Pixal3D / TRELLIS / Hunyuan3D were all trained on neutral-gray renders. `black` bleeds into the PBR atlas and bakes thin black lines along UV seams. Use `white` only for light-on-dark subjects. |

### Outputs

| Slot | Type | Meaning |
|---|---|---|
| `mesh` | TRIMESH | Textured trimesh (PBR materials baked). Chain into TRELLIS2 nodes for preview / export. |
| `preprocessed_image` | IMAGE | Post-bg-removal, post-crop, post-composite image fed to the pipeline. Useful for QA. |
| `camera_angle_x` | FLOAT | FOV (radians) used. Save for cross-image consistency. |
| `distance` | FLOAT | Camera distance used. Pair with `camera_angle_x` to fully describe the camera. |

### Side effects

- A timestamped GLB is **always** written to `ComfyUI/output/pixal3d_<unix_ns>_<seed>.glb`. PNG textures (not WebP) for max viewer compatibility.
- Sets `os.environ["HF_ENDPOINT"]` for the whole process. Required to work around ComfyUI Desktop's inherited mirror env. Affects every HF-aware node in the same worker.

---

## `Pixal3D: Load Pipeline`

Forces pipeline + 4× DINOv3 + MoGe + BiRefNet to load into VRAM. Returns a small token on `pipeline_ready` you can connect to the matching socket on `Image to Mesh` to make warm-up explicit.

| Widget | Default | Notes |
|---|---|---|
| `low_vram` | `false` | Forwarded to the pipeline. |

| Output | Type |
|---|---|
| `pipeline_ready` | PIXAL3D_PIPELINE_READY |

**When to use it:** before queuing a batch of generations, so the 30-60s load isn't hidden inside the first ImageToMesh node. Otherwise unnecessary — ImageToMesh self-bootstraps.

---

## `Pixal3D: Free Pipeline`

Drops the cached pipeline + MoGe singletons and calls `torch.cuda.empty_cache()`. Frees ~14-16 GB VRAM.

No widgets, no inputs, no outputs.

**When to use it:** before switching to a heavy non-Pixal3D workflow on a tight-VRAM card. On a 24 GB+ card you typically don't need this.
