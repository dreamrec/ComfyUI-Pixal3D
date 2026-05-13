# ComfyUI-Pixal3D

[![Pixal3D](https://img.shields.io/badge/Pixal3D-SIGGRAPH%202026-blue)](https://github.com/TencentARC/Pixal3D)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-orange)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT (wrapper)](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pixal3D License: Academic + non-EU](https://img.shields.io/badge/Pixal3D-academic%20%2F%20no--EU-red)](NOTICE.md)
[![Tested on](https://img.shields.io/badge/Tested-Windows%2011%20%2B%20RTX%205090-success)]()

ComfyUI custom node wrapping **[Pixal3D](https://github.com/TencentARC/Pixal3D)**
— Tencent's SIGGRAPH 2026 pixel-aligned image-to-3D pipeline — on **Windows**
with **RTX 30/40/50-series** GPUs. Built on top of the excellent
[ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) (whose
`comfy-env` worker plumbing and CUDA wheel stack we reuse).

A single image → a textured PBR GLB in roughly **3-5 minutes** at default
1024-cascade settings on an RTX 5090.

<p align="center">
  <img src="docs/images/demo_4view.png" alt="Pixal3D 4-view output mesh" width="640">
</p>

> Single-image input on the left of the workflow, four-view turntable
> rendered from the resulting GLB on the right. 152k vertices, 187k faces,
> PBR (base color + metallic + roughness) materials.

---

## ⚠️ Read this before installing

This plugin orchestrates a binary distribution of Pixal3D from Tencent.
**Pixal3D's license is not MIT.** Its terms permit redistribution but with
two strong restrictions you inherit by installing this plugin:

- **Academic / non-commercial use only.** No commercial or production use.
- **NOT for use within the European Union.** Tencent's LICENSE has this clause
  in capitals: *"Pixal3D IS NOT INTENDED FOR USE WITHIN THE EUROPEAN UNION."*

See [`NOTICE.md`](NOTICE.md) for the full third-party license breakdown.

---

## What's in the box

```
ComfyUI-Pixal3D/
├── nodes/                 — three ComfyUI nodes (LoadPipeline, ImageToMesh, FreePipeline)
├── patches/               — small text patches applied to upstream cached files
├── wheels/                — locally compiled natten 0.21.0 wheel (Windows, sm_89 + PTX)
├── workflows/             — example ComfyUI workflow JSONs
├── docs/                  — build instructions for natten + screenshots
├── examples/              — sample input image
├── install.py             — orchestrates the entire install
├── prestartup_script.py   — extends comfy-env worker timeout to 1 hour
├── __init__.py            — ComfyUI node registration
├── requirements.txt       — extra Python deps (moge, utils3d, pyrender)
├── LICENSE                — MIT for the wrapper code
└── NOTICE.md              — upstream license breakdown
```

---

## Hardware + software prerequisites

| | Required | Why |
|---|---|---|
| GPU | NVIDIA RTX 30 / 40 / 50-series with ≥ 16 GB VRAM (24 GB recommended) | Pixal3D's `1024_cascade` peaks ~12 GB VRAM during sampling |
| Disk | ~50 GB free | 24 GB Pixal3D + 1.2 GB DINOv3 + 1.3 GB MoGe + 0.5 GB BiRefNet + 1 GB other |
| OS | Windows 10 / 11 (x64) | tested. Linux works too — install the official natten wheel instead of ours |
| ComfyUI | [Desktop](https://www.comfy.org/download) or portable, recent build | needs `comfy_env` plugin runtime |
| [ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) | installed as a **sibling** custom node | we reuse its pixi-managed worker venv (o_voxel, cumesh, flex_gemm, nvdiffrast, flash_attn) |
| Git | any recent version | `install.py` uses it |
| Network | to reach huggingface.co + github.com | model downloads + Pixal3D source |

---

## Install

> ⚠️ **Before you install**, skim the [Wheel compatibility](#wheel-compatibility)
> table below to confirm the bundled natten wheel will work for your CPU, GPU,
> Python, and PyTorch combo. **Intel and AMD CPUs are both fine** — the wheel
> targets x86_64 generically. The constraints are GPU vendor (NVIDIA only),
> compute capability (sm_50+), Python (3.12 only), and PyTorch (2.8 cu128).

Open PowerShell (or any shell) and `cd` into your ComfyUI custom_nodes folder.
For ComfyUI Desktop the folder is typically:

```powershell
cd "$HOME\Documents\ComfyUI\custom_nodes"
```

1. **Ensure ComfyUI-Trellis2 is installed and has been launched once** (so its
   pixi environment is bootstrapped at `C:\ce\_env_<hash>\.pixi\envs\default\`).
   Easiest way: install it through ComfyUI Manager, restart ComfyUI Desktop,
   wait for "Starting server" in the log.

2. Clone this repo as a **sibling** of ComfyUI-Trellis2:

   ```powershell
   git clone https://github.com/<your-fork>/ComfyUI-Pixal3D.git
   ```

3. Run the installer. It auto-detects the ComfyUI-Trellis2 worker python:

   ```powershell
   cd ComfyUI-Pixal3D
   python install.py
   ```

   What it does, in order:
   - Locates `ComfyUI-Trellis2` next door and finds its pixi-managed Python.
   - `git clone`s [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D)
     at `master` into `_pixal3d_src/` (we don't redistribute Tencent's code).
   - `pip install --no-deps` the extras listed in `requirements.txt`
     (MoGe, utils3d, pyrender, PyOpenGL + accelerate).
   - `pip install --no-deps --force-reinstall` the bundled natten wheel from
     `wheels/`.
   - Applies the BiRefNet inference-mode patch to Pixal3D's `rembg/BiRefNet.py`.
   - Smoke-tests `import natten; import pixal3d.pipelines; import moge.model.v2`.

   Expected total runtime: **30 seconds to 2 minutes** (depending on network
   for the clone + git LFS check). No models are downloaded yet.

4. **Restart ComfyUI Desktop.** The persistent worker has to reload to pick
   up the new node + prestartup patches.

5. Verify the 3 nodes appear in the menu under category **Pixal3D**:

   - `Pixal3D: Load Pipeline` *(optional pre-warm)*
   - `Pixal3D: Image to Mesh` *(the main node)*
   - `Pixal3D: Free Pipeline` *(release VRAM)*

---

## First run

Load `workflows/pixal3d_image_to_mesh.json` from ComfyUI's **Workflow →
Browse** menu (after install + restart). Drop a test image in
`ComfyUI/input/`, point the `LoadImage` node at it, and hit Queue.

**Cold-start downloads on the first queue (one-time, ~26 GB total):**

| Model | Size | HF repo |
|---|---|---|
| Pixal3D weights | ~24 GB | [`TencentARC/Pixal3D`](https://huggingface.co/TencentARC/Pixal3D) |
| DINOv3 ViT-L/16 | ~1.2 GB | [`camenduru/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m) |
| MoGe-2 ViT-L | ~1.3 GB | [`Ruicheng/moge-2-vitl`](https://huggingface.co/Ruicheng/moge-2-vitl) |
| BiRefNet | ~0.44 GB | [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) |

Subsequent runs reuse the HF cache; only inference VRAM is touched.

**Output:** a GLB is **always** written to `ComfyUI/output/pixal3d_<unix_ts>_<seed>.glb`
regardless of any export nodes downstream. Textures are PNG (not WebP) so
Blender / Three.js / etc. open them directly.

---

## The nodes

The plugin registers **3 nodes under the `Pixal3D` category** in the Add Node
menu. Visual reference (defaults shown):

<p align="center">
  <img src="docs/images/nodes_preview.png" alt="Pixal3D nodes preview" width="100%">
</p>

In practice you only need the middle one (`Image to Mesh`); the others are for
explicit warm-up and VRAM cleanup.

### `Pixal3D: Image to Mesh` — the main node

Single image → textured PBR trimesh. Internally chains: optional alpha
attach → BiRefNet background removal (if no mask supplied) → smart crop +
composite → MoGe-2 camera intrinsics → 4× DinoV3 projection-conditioning →
SS / shape-LR / shape-HR / tex-LR / tex-HR sampling cascade → mesh extraction
+ PBR texture bake + decimation + remeshing + GLB export.

#### Required inputs

| Name | Type | Default | Notes |
|---|---|---|---|
| `image` | IMAGE | — | Single image. First batch entry is used. |
| `pipeline_type` | enum | `1024_cascade` | `1024_cascade` (recommended) or `1536_cascade` (heavier, more VRAM, marginally higher fidelity). |
| `seed` | INT | `42` | Same seed + same image + same params → identical output. |

#### Optional inputs (sockets — connect from upstream nodes)

| Name | Type | What | Notes |
|---|---|---|---|
| `mask` | MASK | foreground matte (white = subject) | When provided, BiRefNet is **skipped** entirely. Pipeline attaches your mask as alpha and goes straight to smart-crop + composite. Recommended for complex compositions; pair with RemBG / RMBG-2.0 / SAM. |
| `pipeline_ready` | PIXAL3D_PIPELINE_READY | output of `Pixal3D: Load Pipeline` | Lets you make warm-up explicit (timing visible in its own node). When connected, this node uses the pre-loaded pipeline's `low_vram` setting and ignores the widget. |

#### Optional inputs (widgets — parameters set inline)

##### Pipeline + VRAM

| Name | Type | Default | Range | What it does |
|---|---|---|---|---|
| `low_vram` | BOOLEAN | `false` | — | Offloads sub-models to CPU between stages. Only flip on for cards < 16 GB VRAM. Significantly slower (~2× wall time) because of repeated CPU↔GPU transfers. Ignored if `pipeline_ready` is connected. |
| `max_num_tokens` | INT | `49152` | 16384 – 131072, step 1024 | HR token budget for the 1024-resolution cascade. Lower = less VRAM + smaller mesh detail. `49152` ≈ 9 GB activation peak. Drop to `32768` or `24576` for tighter cards. |
| `mesh_scale` | FLOAT | `1.0` | 0.1 – 5.0 | Object physical scale (used for back-projection calculation). Leave at 1.0 unless your input depicts a known-scale object you want preserved. |
| `image_resolution` | INT | `512` | 256 – 2048, step 64 | Internal resolution for MoGe camera-distance calculation. Doesn't affect generation resolution. |

##### Camera intrinsics

| Name | Type | Default | Range | What it does |
|---|---|---|---|---|
| `camera_angle_x_override` | FLOAT | `0.0` | 0.0 – 3.14 (radians) | `0` means auto-estimate via MoGe-2. Set this **and** `distance_override` both > 0 to bypass MoGe entirely (useful when MoGe fails on stylized art or you want consistent camera across a series). |
| `distance_override` | FLOAT | `0.0` | 0.0 – 50.0 | Same idea — `0` = auto. The two override widgets must **both** be > 0 to actually skip MoGe. |

##### Sparse-structure (SS) sampling — Stage 1 of 3

The sparse-structure stage builds a 3D voxel scaffolding from the image. These
are the standard flow-matching sampler controls.

| Name | Type | Default | Range | What it does |
|---|---|---|---|---|
| `ss_guidance_strength` | FLOAT | `7.5` | 1.0 – 20.0 | CFG strength for the SS sampler. Higher = stronger adherence to image, but can over-fit to the silhouette. |
| `ss_sampling_steps` | INT | `12` | 1 – 50 | Steps per Euler integration. 12 is the sweet spot; 4-8 for quick previews; 16-24 for higher fidelity at ~2× cost. |
| `ss_guidance_rescale` | FLOAT | `0.7` | 0.0 – 1.0 | Classifier-free guidance rescale factor (anti-saturation). |
| `ss_rescale_t` | FLOAT | `5.0` | 0.0 – 10.0 | Time-rescaling exponent for the SS sampler schedule. |

##### Shape SLat sampling — Stage 2 of 3

Generates the dense per-voxel geometry latent (shape) from the SS scaffolding.

| Name | Type | Default | Range | What it does |
|---|---|---|---|---|
| `shape_guidance_strength` | FLOAT | `7.5` | 1.0 – 20.0 | CFG strength for the shape SLat sampler. |
| `shape_sampling_steps` | INT | `12` | 1 – 50 | Same trade-off as `ss_sampling_steps`. |
| `shape_guidance_rescale` | FLOAT | `0.5` | 0.0 – 1.0 | CFG rescale for shape. |
| `shape_rescale_t` | FLOAT | `3.0` | 0.0 – 10.0 | Time-rescaling for shape sampler. |

##### Texture SLat sampling — Stage 3 of 3

Generates the PBR texture latent (base_color + metallic + roughness + alpha)
conditioned on the shape.

| Name | Type | Default | Range | What it does |
|---|---|---|---|---|
| `tex_guidance_strength` | FLOAT | `1.0` | 1.0 – 20.0 | CFG strength for texture. Default is 1.0 (no CFG) because texture is well-conditioned by shape + image. Raising can sharpen but also induces saturation. |
| `tex_sampling_steps` | INT | `12` | 1 – 50 | Steps for texture sampler. |
| `tex_guidance_rescale` | FLOAT | `0.0` | 0.0 – 1.0 | CFG rescale for texture. |
| `tex_rescale_t` | FLOAT | `3.0` | 0.0 – 10.0 | Time-rescaling for texture sampler. |

##### Mesh extraction + texture baking

These control the *post-sampling* mesh post-processing inside `o_voxel.postprocess.to_glb`.

| Name | Type | Default | Range | What it does |
|---|---|---|---|---|
| `decimation_target` | INT | `200000` | 5000 – 1000000, step 5000 | Target face count after mesh decimation. Higher = denser mesh (slower viewer load, less aggressive UV island packing). 200k is a good default for 3D viewers; drop to 50k–100k for game-engine use. |
| `texture_size` | INT | `2048` | 256 – 4096, step 256 | PBR texture atlas resolution per channel. 2048 is solid for most subjects; 4096 helps when you can see UV seams. Larger = slower bake + bigger GLB. |
| `remesh` | BOOLEAN | `true` | — | Run the o_voxel post-remeshing pass that improves triangle quality. Turn off only if it's introducing artifacts you can identify and traced to remeshing specifically. |
| `background_color` | enum | `gray` | `gray` / `black` / `white` | Color composited behind the foreground mask. **`gray` (128,128,128) is the safe default** — it matches the neutral renders Pixal3D / TRELLIS / Hunyuan3D were trained on and prevents the *thin-black-line bleed* into the PBR atlas that `black` causes. Use `white` only for light-on-dark subjects. |

#### Outputs

| Slot | Type | Meaning |
|---|---|---|
| `mesh` | TRIMESH | Textured trimesh scene (PBR materials baked). Chain into `Trellis2RenderPreview` for an 8-view turntable, `Trellis2ExportTrimesh` to write a custom file, or any other TRIMESH-consuming node. |
| `preprocessed_image` | IMAGE | The post-bg-removal, post-crop, post-composite image the pipeline actually fed itself. Useful for QA — if it looks wrong here, your mesh will look wrong. |
| `camera_angle_x` | FLOAT | The horizontal FOV (radians) MoGe estimated, or your override value. Useful to feed into other camera-aware nodes, or to lock cross-image consistency by reusing this value via the override widgets. |
| `distance` | FLOAT | The camera distance used. Pair with `camera_angle_x` to fully describe the camera in subsequent runs. |

#### Side effects (not exposed via output sockets)

- A timestamped GLB is **always** written to
  `ComfyUI/output/pixal3d_<unix_ns_ts>_<seed>.glb` regardless of what's
  downstream of `mesh`. Textures are PNG (not WebP) so Blender / Three.js /
  Babylon open them directly.
- The plugin **unconditionally sets `os.environ["HF_ENDPOINT"] = "https://huggingface.co"`**
  at module load. This is a process-wide change that affects every other
  HuggingFace-aware node in the same ComfyUI worker. Necessary because
  ComfyUI Desktop has been observed to inherit `HF_ENDPOINT=https://hf-mirror.com`
  from somewhere upstream, blocking BiRefNet / RMBG-2.0 downloads. See the
  comment in `nodes/pixal3d_stages.py` for full context.

---

### `Pixal3D: Load Pipeline` — explicit warm-up

Forces pipeline + 4× DINOv3 extractors + MoGe-2 + BiRefNet to load and reside
on GPU. Returns a small dict on a `PIXAL3D_PIPELINE_READY` socket that you can
optionally feed into `Image to Mesh` to make the warm-up dependency explicit.

| Widget | Type | Default | What it does |
|---|---|---|---|
| `low_vram` | BOOLEAN | `false` | Forwarded to the pipeline. Same semantics as the matching widget on `Image to Mesh`. |

| Output | Type | Meaning |
|---|---|---|
| `pipeline_ready` | PIXAL3D_PIPELINE_READY | A token signaling "the pipeline is in VRAM and ready to sample." Connect to the matching socket on `Image to Mesh` to surface load time in its own node, or omit this node entirely and let `Image to Mesh` load on demand. |

**When to use it:** before a sweep where you'll queue many generations, to
avoid each Queue button click looking frozen for 30-60 s while the pipeline
loads. Otherwise unnecessary — `Image to Mesh` is self-bootstrapping.

---

### `Pixal3D: Free Pipeline` — VRAM cleanup

Drops the cached pipeline + MoGe singletons and calls `torch.cuda.empty_cache()`.
Releases ~14-16 GB of VRAM.

No widgets, no inputs, no outputs.

**When to use it:** before switching to a heavy non-Pixal3D workflow on a
tight-VRAM card. On a 24+ GB card you typically don't need this — Pixal3D's
footprint coexists with most other workflows.

---

## Masking: internal vs external

Pixal3D needs a clean foreground mask before sampling. You have two ways to
provide it.

### Path A — let Pixal3D mask the image itself (simplest)

```
LoadImage ──► Pixal3DImageToMesh ──► GLB
                  (no mask input)
```

When you don't connect a MASK, Pixal3D runs its **own BiRefNet** internally
(rerouted from the gated `briaai/RMBG-2.0` to the ungated
[`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) by this
plugin). One node, no extra setup, no extra dependencies. The
`preprocessed_image` output shows you what mask + crop + composite the
pipeline actually fed itself, useful for QA after the fact.

### Path B — bring your own mask (full control)

```
LoadImage ──┬─► RemBGSession+ ──► ImageRemoveBackground+ ──► MASK ──┐
            └──────────────────────────────────────────────────────┴──► Pixal3DImageToMesh
                                                                       (image + mask)
```

Connect any MASK input — from `comfyui-rmbg` (u2net / BiRefNet / RMBG-2.0),
SAM, manual painting, an `INVERT_MASK` node, etc. When a mask is present
Pixal3D attaches it as the alpha channel and **completely skips its internal
BiRefNet**, going straight to smart-crop + composite. See
`workflows/pixal3d_image_to_mesh_with_external_rembg.json` for a working
example using comfyui-rmbg's u2net.

### When to pick which

| Path | Use when |
|---|---|
| **A (internal)** | Simple subject, clean background, you trust BiRefNet's matte (it's good for most photos / single objects) |
| **B (external)** | Complex composition (floating islands with overhanging elements, multiple subjects), need to *see* the mask before committing 3-5 min of generation, iterating on the same image, or you want a different masker (RMBG-2.0, SAM with prompts, etc.) |

Either path produces the same `Pixal3DImageToMesh` outputs. Pixal3D's own
BiRefNet patch (`patches/birefnet_inference_mode.py`) is required for Path A
to work on Windows + recent PyTorch; the installer applies it for you. If you
use a third-party BiRefNet node in Path B that hits the same *"Inference
tensors do not track version counter"* error, apply the same one-line fix
(`torch.no_grad()` → `torch.inference_mode()`) to that node's code.

## Example workflows

### `workflows/pixal3d_image_to_mesh.json` — minimal

```
LoadImage ──► Pixal3DImageToMesh ──┬─► Trellis2RenderPreview ──► PreviewImage  (8-view turntable)
                                   └─► PreviewImage  (preprocessed input)
                                   └─► GLB auto-saved to output/
```

### `workflows/pixal3d_image_to_mesh_with_external_rembg.json` — recommended

Adds a `RemBGSession+` (u2net via CUDA) → `ImageRemoveBackground+` chain that
feeds a MASK into the Pixal3D node. This bypasses the built-in BiRefNet
preprocessing entirely (so the Pixal3D pipeline goes straight to smart-crop
+ composite), and gives you a `MaskPreview` of the cutout for QA. Works great
on complex compositions with cluttered backgrounds.

---

## Settings cheat-sheet

| Goal | Suggested settings |
|---|---|
| Default quality (~3-5 min) | `1024_cascade`, all sampling steps = 12, max_num_tokens = 49152 |
| Fast preview (~1.5 min) | `1024_cascade`, sampling steps = 4, max_num_tokens = 32768, decimation = 100k |
| Maximum fidelity (~8-12 min) | `1536_cascade`, sampling steps = 16, max_num_tokens = 65536, decimation = 500k, texture_size = 4096 |
| < 16 GB VRAM | `low_vram = True`, max_num_tokens = 24576, sampling steps = 8 |

---

## Troubleshooting

### `RuntimeError: Inference tensors do not track version counter.`
The `patches/birefnet_inference_mode.py` patch hasn't been applied. Re-run
`python install.py` (it's idempotent) then restart ComfyUI Desktop.

### `No module named 'pyrender'` (or `pyopengl`)
Worker env missing the render preview deps. Re-run `python install.py`. If
pyrender errors during render with `No array-type handler for type _ctypes.type`,
you have stale PyOpenGL 3.1.0 — install upgrades `PyOpenGL` and
`PyOpenGL-accelerate` to ≥ 3.1.7.

### `OSError: We couldn't connect to 'https://hf-mirror.com'`
Your environment has `HF_ENDPOINT` set to the Chinese mirror. The plugin
overrides this internally, but make sure you also don't have a stale value
visible to ComfyUI Desktop's process. Restart after install fixes it.

### `Allocation on device` / `OutOfMemoryError`
Drop `max_num_tokens` to 32768 or 24576, drop `pipeline_type` to `1024_cascade`,
and/or enable `low_vram`. With 16 GB you usually need at least two of those.

### Thin black lines / dark seams in the textured mesh
Caused by Pixal3D's `preprocess_image` compositing the foreground onto a
**pure-black** background by default — dark pixels bleed into the PBR texture
atlas at the silhouette and reappear as thin lines across UV seams. The
`Pixal3D: Image to Mesh` node exposes a `background_color` widget which now
defaults to `"gray"` (128, 128, 128). Keep it on `gray` unless your subject
is very light against very dark — only then try `white` or `black`. If you
already have an existing-bad GLB, re-running with `background_color=gray`
fixes it (the seams come from the texture, not the geometry).

### `Failed opening glTF file: Unknown image format. STB cannot decode image data`
Older saved GLBs may have WebP textures (legacy default copied from Pixal3D's
inference.py). New saves are PNG. Re-load an existing GLB and re-export:
```python
import trimesh
m = trimesh.load("old.glb")
m.export("new_png.glb")  # default = PNG
```

### `comfy_env.isolation.workers.base.WorkerError: TimeoutError`
Means `pipeline.run()` took longer than the per-call worker timeout. Our
`prestartup_script.py` raises this to 3600s for `Pixal3D*` classes. If it
still triggers, drop sampling steps and `max_num_tokens`. The prestartup
script only takes effect after a ComfyUI restart.

---

## Wheel compatibility

> Read this before installing if your machine isn't a stock ComfyUI-Trellis2
> + RTX 50-series Windows box like the one this was built on.

The bundled `wheels/natten-0.21.0+winsm89ptx-cp312-cp312-win_amd64.whl` is a
**locally-compiled** binary. Each tag in the filename locks down a different
constraint:

| Constraint | Locked to | Works on | Does NOT work on |
|---|---|---|---|
| **CPU vendor** | `win_amd64` = 64-bit x86 | Any **Intel** *or* **AMD** x86_64 CPU — they run the exact same instruction set | ARM (Snapdragon X, Surface Pro X, Apple Silicon) |
| **CPU architecture** | x86_64 (AMD64) | Modern Intel Core (4th gen+) / Xeon and AMD Ryzen / Threadripper / EPYC | 32-bit x86, ARM64, RISC-V |
| **OS** | `win_amd64` | Windows 10, 11 (x64) | Linux (needs `.so`), macOS (needs `.dylib`), Windows on ARM |
| **Python ABI** | `cp312` | CPython **3.12.x** only | 3.10, 3.11, 3.13, PyPy, etc. — strict mismatch |
| **GPU vendor** | linked against CUDA | **NVIDIA** | AMD ROCm, Intel Arc, Apple Metal |
| **GPU compute capability** | sm_89 SASS + PTX fallback | Native fast: RTX 4070 / 4080 / 4090 (sm_89). PTX-JITted (works, slightly slower on first launch only): RTX 20 / 30 / 40 / 50 series, A-series, T4, V100, H100, B100/B200. Anything sm_50 through sm_120. | Pre-Maxwell (GTX 900 series and older, sm_30) |
| **CUDA runtime** | linked against CUDA 12.8 | NVIDIA driver ≥ 555 (CUDA 12.8 forward-compat shim covers later drivers too) | Drivers older than ~525 (CUDA 12.0) |
| **PyTorch ABI** | torch 2.8.0+cu128 | torch 2.8.x with cu128 | torch 2.6 / 2.7 / 2.9 / 2.10 (possible C++ ABI breaks); torch with cu118 / cu121 / cu124 (definitely broken) |

### Practical bottom line — will it work for me?

**If your friend runs the recommended setup** — ComfyUI Desktop on Windows
10/11 x64 with [ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2)
already working (its pixi env always pins torch 2.8 cu128 + Python 3.12) —
**the wheel just works** on their box, regardless of:

- ✅ Intel Core i5 / i7 / i9 of any recent generation
- ✅ Intel Xeon
- ✅ AMD Ryzen 5 / 7 / 9, Threadripper, EPYC
- ✅ Combined with any NVIDIA RTX 20 / 30 / 40 / 50 series, A-series, etc.

**Where the wheel does NOT work** and they need to rebuild via
[`docs/BUILD_NATTEN.md`](docs/BUILD_NATTEN.md) (or use an upstream wheel):

- ❌ **Linux**: skip `install.py`'s wheel install; use the official Linux
  wheel instead — `pip install natten==0.21.0+torch270cu128 -f https://whl.natten.org`
  (or whichever torch+cuda matches your env). Everything else — the Python
  deps, the BiRefNet patch, the nodes themselves — is identical.
- ❌ **Python ≠ 3.12** (e.g. portable ComfyUI on 3.11 or 3.13): rebuild for
  your Python version. The `cp312` tag is strict; pip will refuse to install
  the wheel.
- ❌ **PyTorch ≠ 2.8.x + cu128**: rebuild against your torch/cuda combo.
  Most relevant if you've manually upgraded torch inside the pixi env.
- ❌ **ARM Windows** / **Apple Silicon** / **WSL2 ARM**: not supported.
- ❌ **AMD or Intel GPUs**: not supported by Pixal3D at all — upstream
  limitation, not ours.

### Sanity-check after install

After `python install.py` completes, run this from any shell:

```powershell
& '<worker_python>' -c "import natten, torch; q = torch.randn(1,32,32,4,64,device='cuda',dtype=torch.float16); v = torch.randn(1,32,32,4,256,device='cuda',dtype=torch.float16); print('HAS_LIBNATTEN', natten.HAS_LIBNATTEN, '|', natten.na2d(q,q,v,kernel_size=(9,9),backend='cutlass-fna').shape)"
```

Expected output: `HAS_LIBNATTEN True | torch.Size([1, 32, 32, 4, 256])`.
If you see `HAS_LIBNATTEN False` or `no kernel image is available for execution`,
your wheel is mismatched — rebuild per `docs/BUILD_NATTEN.md`.

---

## Credits

This plugin is just plumbing. The actual research / model work belongs to:

- **[Pixal3D](https://github.com/TencentARC/Pixal3D)** — Li, Zhao, Chen, Hu,
  Guo, Zhang, Shan & Hu (Tsinghua + Tencent ARC), SIGGRAPH 2026.
- **[TRELLIS.2](https://github.com/microsoft/TRELLIS.2)** — Microsoft Research.
- **[ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2)** —
  visualbruno, whose `comfy_env` integration and CUDA wheel stack we ride on.
- **[NATTEN](https://github.com/SHI-Labs/NATTEN)** — SHI-Labs / Ali Hassani.
- **[NAF](https://github.com/valeoai/NAF)** — valeoai.
- **[MoGe](https://github.com/microsoft/MoGe)** — Microsoft Research.
- **[BiRefNet](https://github.com/ZhengPeng7/BiRefNet)** — ZhengPeng7.

Wrapper code: dreamrec (2026). MIT — see [`LICENSE`](LICENSE) and
[`NOTICE.md`](NOTICE.md).
