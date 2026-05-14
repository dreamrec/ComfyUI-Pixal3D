# Third-Party Notices

ComfyUI-Pixal3D orchestrates several upstream projects. By installing or using
this plugin you agree to comply with each of their licenses. The most
restrictive is Pixal3D's — read its terms before you publish, sell, or use it
inside the EU.

## Pixal3D — Tencent License (NOT MIT)

Pixal3D is © 2026 Tencent and licensed under Tencent's custom "License Term
of Pixal3D" (see https://github.com/TencentARC/Pixal3D/blob/master/LICENSE).
Key clauses you inherit by using this plugin:

- **Academic / non-commercial use only.** No commercial or production use.
- **NOT for use within the European Union.** Tencent's LICENSE explicitly
  states *"Pixal3D IS NOT INTENDED FOR USE WITHIN THE EUROPEAN UNION."*
- The Tencent copyright + permission notice must travel with anything you
  redistribute that contains Pixal3D code or weights.

This plugin does **not** redistribute Pixal3D's source — the installer
clones it from Tencent's official GitHub at install time. Pixal3D model
weights are downloaded from Hugging Face (`TencentARC/Pixal3D`) on first
node run.

## Other upstream dependencies

| Project | Role | License |
|---|---|---|
| [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) | image-to-3D pipeline + weights | Tencent custom (see above) |
| [SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN) | neighborhood-attention CUDA kernels (we bundle a locally compiled wheel) | MIT |
| [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | upstream backbone (via the [`ComfyUI-Trellis2`](https://github.com/visualbruno/ComfyUI-Trellis2) wrapper) | MIT |
| [valeoai/NAF](https://github.com/valeoai/NAF) | per-pixel feature upsampler used inside Pixal3D | Apache 2.0 |
| [microsoft/MoGe](https://github.com/microsoft/MoGe) | camera-intrinsics estimator | MIT |
| [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) | background-removal model (rerouted from gated briaai/RMBG-2.0) | MIT |
| facebookresearch DINOv3 (via `camenduru/dinov3-vitl16-pretrain-lvd1689m`) | image conditioning encoder | Meta Research license — see HF repo |
| [LDYang694/utils3d](https://github.com/LDYang694/Storages) | minor 3D utilities (`utils3d-0.0.2-py3-none-any.whl`) | per-repo (no LICENSE file as of release 20260430) |

## Bundled binary

`wheels/natten-0.21.0+winsm89ptx-cp312-cp312-win_amd64.whl` is a locally compiled
build of NATTEN 0.21.0 (MIT, © 2022–2025 Ali Hassani) patched for MSVC + CUDA
12.8 + RTX 50-series PTX forward-compat. The wheel embeds its own LICENSE in
its metadata. Our build patches are visible in `patches/natten_msvc.diff`.

## What this repository contains

Original work in this repo (under the MIT LICENSE):

- `nodes/` — ComfyUI node classes that bridge Pixal3D into a ComfyUI workflow.
- `install.py` — orchestration script that clones upstream + applies patches.
- `patches/` — small text patches against upstream cache files for Windows
  compatibility (BiRefNet inference-mode, NAF flex/cutlass backend).
- `prestartup_script.py` — extends ComfyUI's per-call worker timeout.
- `workflows/` — example ComfyUI workflow JSONs.
- `wheels/` — the locally compiled natten wheel.
