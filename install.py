"""
ComfyUI-Pixal3D installer.

Steps (idempotent — safe to re-run):
  1. Locate the sibling ComfyUI-Trellis2 custom node (we reuse its pixi env).
  2. Resolve the worker Python executable from ComfyUI-Trellis2's comfy-env config.
  3. Clone TencentARC/Pixal3D@master into `_pixal3d_src/` (skip if present).
  4. `pip install --no-deps` the contents of `requirements.txt`.
  5. `pip install --no-deps` the bundled natten wheel from `wheels/`.
  6. Apply the BiRefNet inference-mode patch.
  7. Sanity-import pixal3d + natten + moge + utils3d in the worker env.

Run from inside `custom_nodes/ComfyUI-Pixal3D/`:

    "<worker_python>" install.py

The script auto-detects the worker python; if detection fails it prints a
clear error and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import List, Optional

HERE = pathlib.Path(__file__).resolve().parent
PIXAL3D_REPO = "https://github.com/TencentARC/Pixal3D.git"
# Pinned to the exact commit verified working with this plugin. Update only
# after re-testing against a newer upstream — Tencent has force-pushed
# `master` in the past, so a moving target breaks reproducibility.
PIXAL3D_REF = "d42dcaad99ba07d35a02fa62a23e1cd6f2f61da1"


def log(msg: str) -> None:
    print(f"[ComfyUI-Pixal3D install] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Worker env discovery
# ----------------------------------------------------------------------------

def find_trellis2_root() -> pathlib.Path:
    """The TRELLIS2 custom node sits as a sibling of this repo."""
    candidate = HERE.parent / "ComfyUI-TRELLIS2"
    if candidate.is_dir():
        return candidate
    raise SystemExit(
        f"Could not find ComfyUI-Trellis2 next to {HERE}. "
        "Install https://github.com/visualbruno/ComfyUI-Trellis2 first."
    )


def find_worker_python(trellis2_root: pathlib.Path) -> pathlib.Path:
    """Look for the pixi-managed python that comfy-env created for TRELLIS2."""
    # 1. Most common: the env_dir is recorded in a `.comfy-env-root.toml`.
    for marker in ("comfy-env-root.toml", ".comfy-env-root.toml"):
        p = trellis2_root / marker
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if "env_dir" in line and "=" in line:
                    raw = line.split("=", 1)[1].strip().strip('"').strip("'")
                    py = pathlib.Path(raw) / ".pixi" / "envs" / "default" / "python.exe"
                    if py.is_file():
                        return py

    # 2. Scan typical comfy-env install root.
    fallback_root = pathlib.Path(r"C:\ce")
    if fallback_root.is_dir():
        candidates = sorted(fallback_root.glob("_env_*/.pixi/envs/default/python.exe"))
        if candidates:
            return candidates[-1]

    raise SystemExit(
        f"Could not locate the ComfyUI-Trellis2 worker python. Looked under "
        f"{fallback_root}. Make sure ComfyUI-Trellis2 has been installed and "
        f"its pixi env initialised at least once."
    )


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------

def run(cmd: List[str], cwd: Optional[pathlib.Path] = None) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], cwd=cwd)


def clone_pixal3d() -> None:
    target = HERE / "_pixal3d_src"
    if target.is_dir() and (target / "pixal3d").is_dir():
        log(f"Pixal3D source already at {target} — skipping clone.")
        return
    if shutil.which("git") is None:
        raise SystemExit("git not found on PATH. Install git first.")
    # `clone --depth 1 <sha>` doesn't work on most servers — do a shallow
    # clone of the default branch, then fetch + checkout the pinned SHA.
    run(["git", "clone", "--filter=blob:none", "--no-checkout", PIXAL3D_REPO, str(target)])
    run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", PIXAL3D_REF])
    run(["git", "-C", str(target), "checkout", PIXAL3D_REF])
    log(f"Checked out Pixal3D @ {PIXAL3D_REF[:12]}")


def pip_install(python: pathlib.Path, args: List[str]) -> None:
    run([str(python), "-m", "pip", "install"] + args)


def install_requirements(python: pathlib.Path) -> None:
    req = HERE / "requirements.txt"
    if not req.is_file():
        raise SystemExit(f"requirements.txt not found at {req}")
    # --no-deps to avoid pip resolver fighting with the pixi-managed env.
    pip_install(python, ["--no-deps", "-r", str(req)])


def install_natten_wheel(python: pathlib.Path) -> None:
    wheels = sorted((HERE / "wheels").glob("natten-*.whl"))
    if not wheels:
        raise SystemExit(
            "No natten wheel in wheels/. Build one with docs/BUILD_NATTEN.md "
            "or download a release asset."
        )
    pip_install(python, ["--no-deps", "--force-reinstall", str(wheels[-1])])


def apply_patches() -> None:
    """Call each patch script's `main()` directly so a non-zero return code
    doesn't raise SystemExit through install.py's flow."""
    import importlib.util

    for patch_name in ("birefnet_inference_mode",):
        path = HERE / "patches" / f"{patch_name}.py"
        spec = importlib.util.spec_from_file_location(f"patches.{patch_name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.main()
        if rc != 0:
            log(f"[apply_patches] {patch_name}.main() returned {rc} — continuing.")


def sanity_check(python: pathlib.Path) -> None:
    code = (
        "import sys, os; "
        f"sys.path.insert(0, r'{HERE / '_pixal3d_src'}'); "
        "os.environ.setdefault('ATTN_BACKEND', 'flash_attn'); "
        "import natten; "
        "import moge.model.v2; "
        "import utils3d; "
        "from pixal3d.pipelines import Pixal3DImageTo3DPipeline; "
        "from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor; "
        "print('natten', natten.__version__, 'HAS_LIBNATTEN=', natten.HAS_LIBNATTEN); "
        "print('OK')"
    )
    run([str(python), "-c", code])


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        help="Path to the worker python (auto-detected if omitted).",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Don't clone Pixal3D source (use if you've supplied your own _pixal3d_src/).",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Don't pip-install requirements.txt or the natten wheel.",
    )
    args = parser.parse_args()

    trellis2 = find_trellis2_root()
    log(f"ComfyUI-Trellis2 at: {trellis2}")

    worker_py = pathlib.Path(args.python) if args.python else find_worker_python(trellis2)
    log(f"Worker python: {worker_py}")

    if not args.skip_clone:
        clone_pixal3d()
    if not args.skip_deps:
        install_requirements(worker_py)
        install_natten_wheel(worker_py)
    apply_patches()
    sanity_check(worker_py)

    log("")
    log("Install complete. Restart ComfyUI Desktop, then load a workflow from "
        "workflows/. The first run downloads ~26 GB of model weights to the "
        "HuggingFace cache; subsequent runs reuse them and complete in ~3-5 min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
