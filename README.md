# ComfyUI-Pixal3D

[![Pixal3D](https://img.shields.io/badge/Pixal3D-SIGGRAPH%202026-blue)](https://github.com/TencentARC/Pixal3D)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-orange)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT (wrapper)](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pixal3D License](https://img.shields.io/badge/Pixal3D-academic%20%2F%20no--EU-red)](NOTICE.md)

ComfyUI custom node for **[Pixal3D](https://github.com/TencentARC/Pixal3D)** — Tencent's SIGGRAPH 2026 single-image to PBR-textured-3D pipeline — on **Windows** with **RTX 30/40/50** GPUs. Built on top of [ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2).

**One image → textured PBR mesh in ~3-5 min on an RTX 5090.**

<p align="center">
  <img src="docs/images/demo_4view.png" alt="Pixal3D output mesh — 4 views" width="640">
</p>

---

## ⚠️ License you inherit

Pixal3D is licensed by Tencent for **academic / non-commercial use only**, and **explicitly NOT for use within the European Union**. By installing this plugin you agree to those terms. See [NOTICE.md](NOTICE.md).

---

## Will it work on my machine?

**Short version:** if you have ComfyUI Desktop running TRELLIS2 successfully on an RTX 30/40/50 GPU, this will work too.

| | Requirement |
|---|---|
| OS | Windows 10 / 11 (x64) — bundled natten wheel is Windows-only |
| GPU | NVIDIA RTX 30 / 40 / 50 with ≥ **16 GB VRAM** (24 GB+ recommended for `1024_cascade` defaults) |
| Disk | ~50 GB free (24 GB Pixal3D + 4 GB other models + workspace) |
| CPU | Any modern x86_64 — **Intel and AMD both work**, no special requirements |
| Python | **3.12 only** (your worker venv must be 3.12 — the bundled wheel is cp312) |
| PyTorch | **2.8.x + CUDA 12.8** (your worker venv must match — wheel is built against torch 2.8.0+cu128) |
| ComfyUI | Desktop (or portable) with **[ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) already installed and launched once** |

### Will the wheel work for me?

The bundled `wheels/natten-0.21.0+winsm89ptx-...-win_amd64.whl` is locked to **Windows + Python 3.12 + PyTorch 2.8 + CUDA 12.8 + NVIDIA GPU**. If your setup matches all of the above (which the standard ComfyUI-Trellis2 pixi env does), the wheel just works.

If any one of those doesn't match (you're on Linux / Python 3.11 / PyTorch 2.7 / etc.) you need to **rebuild natten for your env** — see [docs/BUILD_NATTEN.md](docs/BUILD_NATTEN.md). On Linux you can also just `pip install natten==0.21.0 -f https://whl.natten.org` to use the official prebuilt wheel.

**AMD / Intel GPUs are not supported** — upstream Pixal3D requires CUDA.

---

## Prerequisites — TRELLIS2 must be working first

This plugin **does not** install its own CUDA stack. It rides on top of ComfyUI-Trellis2's pixi-managed worker environment (which already contains o_voxel, cumesh, flex_gemm, nvdiffrast, flash_attn, etc.).

**Before installing this plugin:**

1. Install **[ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2)** via ComfyUI Manager (or git clone into `custom_nodes/`).
2. **Launch ComfyUI Desktop once** — TRELLIS2's first launch bootstraps its pixi env at `C:\ce\_env_<hash>\.pixi\envs\default\`. You'll see "Starting server" in the log when it's ready.
3. **Verify TRELLIS2 works** — load one of its example workflows and queue it. If TRELLIS2 itself errors, fix that first; Pixal3D won't help you.

If you skip these, the Pixal3D installer can't find the worker Python and will exit with a clear error message.

---

## Install

```powershell
# 1. Open the custom_nodes folder
cd $HOME\Documents\ComfyUI\custom_nodes

# 2. Clone this repo *next to* ComfyUI-Trellis2 (NOT inside it)
git clone https://github.com/dreamrec/ComfyUI-Pixal3D.git

# 3. Run the installer
cd ComfyUI-Pixal3D
python install.py

# 4. Restart ComfyUI Desktop
```

What `install.py` does, in ~30 seconds:

- Auto-detects the TRELLIS2 worker Python.
- Clones [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) at a pinned commit into `_pixal3d_src/`.
- Installs MoGe + utils3d + pyrender + PyOpenGL into the worker venv.
- Installs the bundled natten wheel from `wheels/`.
- Patches Pixal3D's BiRefNet for the Windows `inference_mode` interaction.
- Sanity-checks all imports.

**On first queue**, ComfyUI will download ~26 GB of model weights from HuggingFace (one-time, cached): Pixal3D weights (24 GB) + DINOv3 (1.2 GB) + MoGe-2 (1.3 GB) + BiRefNet (0.44 GB). Cold-start with download takes ~30 min on a fast connection; subsequent runs use the cache.

---

## Use

After install + restart, three nodes appear in the Add Node menu under **`Pixal3D`**:

<p align="center">
  <img src="docs/images/nodes_preview.png" alt="Pixal3D nodes" width="100%">
</p>

The only one you need is **`Pixal3D: Image to Mesh`**. Drop in an image, queue, get a GLB.

### Two workflows are bundled in `workflows/`

| File | Use when |
|---|---|
| `pixal3d_image_to_mesh.json` | Default. Internal BiRefNet does background removal automatically. |
| `pixal3d_image_to_mesh_with_external_rembg.json` | You want a better matte than BiRefNet (RMBG-2.0, SAM, manual). Connects a mask into the node, which skips internal background removal. |

Load either via **Workflow → Browse**. Drop your image into the LoadImage node, hit Queue.

GLBs are auto-saved to `ComfyUI/output/pixal3d_<timestamp>_<seed>.glb` with PNG-textures (open in Blender / Three.js / any standard viewer).

**Full parameter reference** for all three nodes lives in [docs/NODES.md](docs/NODES.md).

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Could not find ComfyUI-Trellis2` during install | ComfyUI-Trellis2 isn't installed next to this repo, or it hasn't been launched once to bootstrap its pixi env. Install/launch TRELLIS2 first. |
| `Inference tensors do not track version counter` mid-run | You're on an older version of this plugin. `git pull` and re-run `python install.py`. (Fixed by wrapping `run_pixal3d` in `torch.inference_mode(False)`.) |
| Thin black lines on the textured mesh | Set the `background_color` widget on the node to `gray` (the default). If you're on an old saved workflow it may still have `black` — re-create the node from the menu. |
| `No module named 'pyrender'` or PyOpenGL ctypes error | Worker venv missing render deps. Re-run `python install.py`. |
| `OSError: We couldn't connect to 'https://hf-mirror.com'` | Your environment has `HF_ENDPOINT` set to the Chinese mirror. The plugin overrides this internally; if you still hit it, restart ComfyUI after install. |
| `OutOfMemoryError` / `Allocation on device` | Drop `max_num_tokens` to 32768 or 24576, optionally enable `low_vram` on the node. |
| Blender refuses to open the GLB (`STB cannot decode image data`) | You have an old GLB from before this fix. Re-run; new GLBs use PNG textures. |
| Workflow JSON rejected with widget-index errors | You saved the workflow from an old plugin version. Delete the `Pixal3DImageToMesh` node and add a fresh one from the menu. |

---

## Memory + performance

| Setting | VRAM peak | Time | Quality |
|---|---|---|---|
| `1024_cascade` + steps 12/12/12 (defaults) | ~12 GB | ~3-5 min | Recommended |
| `1024_cascade` + steps 4/4/4 + max_tokens 32768 | ~7 GB | ~1.5 min | Preview |
| `1536_cascade` + steps 16/16/16 + tex_size 4096 | ~22 GB | ~8-12 min | Maximum |
| `low_vram=true` + max_tokens 24576 | ~8 GB | ~6-8 min | Tight cards |

---

## Credits + license

Wrapper code (this repo): **MIT**, dreamrec 2026.

The actual research / model work belongs to:

- **[Pixal3D](https://github.com/TencentARC/Pixal3D)** — Tencent ARC + Tsinghua, SIGGRAPH 2026. **Tencent license — academic only, no EU use.**
- **[TRELLIS.2](https://github.com/microsoft/TRELLIS.2)** + **[ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2)** — Microsoft Research + visualbruno (MIT).
- **[NATTEN](https://github.com/SHI-Labs/NATTEN)** — SHI-Labs (MIT).
- **[NAF](https://github.com/valeoai/NAF)** — valeoai (Apache 2.0).
- **[MoGe](https://github.com/microsoft/MoGe)** — Microsoft Research (MIT).
- **[BiRefNet](https://github.com/ZhengPeng7/BiRefNet)** — ZhengPeng7 (MIT).

Full third-party license breakdown in [NOTICE.md](NOTICE.md).
