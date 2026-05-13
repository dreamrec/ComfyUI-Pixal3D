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


def natten_already_works(python: pathlib.Path) -> bool:
    """Probe the worker python: do we have natten installed AND with the
    libnatten C extension working AND able to run na2d on cuda?

    Returns True only if all three hold — i.e. the user has either already
    run this installer before OR built+installed their own natten wheel
    (per docs/BUILD_NATTEN.md). In that case install.py should NOT
    overwrite their natten with the bundled wheel.
    """
    probe = (
        "import sys\n"
        "try:\n"
        "    import natten, torch\n"
        "    if not torch.cuda.is_available():\n"
        "        print('NO_CUDA'); sys.exit(0)\n"
        "    if not natten.HAS_LIBNATTEN:\n"
        "        print('NO_LIBNATTEN'); sys.exit(0)\n"
        "    q = torch.randn(1,8,8,4,64,device='cuda',dtype=torch.float16)\n"
        "    v = torch.randn(1,8,8,4,256,device='cuda',dtype=torch.float16)\n"
        "    natten.na2d(q,q,v,kernel_size=(3,3),backend='cutlass-fna')\n"
        "    print('OK', natten.__version__)\n"
        "except Exception as _e:\n"
        "    print('FAIL', type(_e).__name__, str(_e)[:120])\n"
    )
    try:
        out = subprocess.check_output(
            [str(python), "-c", probe], stderr=subprocess.STDOUT, timeout=120
        ).decode("utf-8", errors="replace").strip()
    except subprocess.CalledProcessError as e:
        out = (e.output or b"").decode("utf-8", errors="replace").strip()
    except Exception as e:
        out = f"PROBE_ERROR {type(e).__name__} {e}"
    # The probe may emit unrelated warnings on stderr (e.g.
    # "expandable_segments not supported on this platform") that come BEFORE
    # the print(). Scan all non-empty lines for the result marker — the
    # probe's print is the authoritative line, anything else is noise.
    verdict_line = ""
    for ln in reversed(out.splitlines()):
        s = ln.strip()
        if s.startswith(("OK ", "FAIL ", "NO_CUDA", "NO_LIBNATTEN", "PROBE_ERROR")):
            verdict_line = s
            break
    log(f"[natten probe] {verdict_line or out}")
    return verdict_line.startswith("OK ")


def install_natten_wheel(python: pathlib.Path) -> None:
    if natten_already_works(python):
        log("natten is already installed and working — keeping it. "
            "(Re-run with --force-natten-wheel to overwrite anyway.)")
        return

    wheels = sorted((HERE / "wheels").glob("natten-*.whl"))
    if not wheels:
        raise SystemExit(
            "natten is not installed AND no wheel found in wheels/. Either:\n"
            "  • re-download this repo (the wheel should be bundled), OR\n"
            "  • build your own wheel via docs/BUILD_NATTEN.md and either\n"
            "    install it manually before re-running this installer,\n"
            "    or drop it into wheels/ and re-run this installer."
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
    parser.add_argument(
        "--skip-natten-wheel",
        action="store_true",
        help="Don't touch natten. Use when you built + installed your own "
             "wheel for a different Python/PyTorch/GPU combo per "
             "docs/BUILD_NATTEN.md. install.py still installs requirements.txt "
             "and applies patches.",
    )
    parser.add_argument(
        "--force-natten-wheel",
        action="store_true",
        help="Install the bundled natten wheel even if a working natten is "
             "already detected. Overwrites a manually built natten.",
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
        if args.skip_natten_wheel:
            log("--skip-natten-wheel: leaving natten alone.")
        elif args.force_natten_wheel:
            log("--force-natten-wheel: bypassing the 'already works' probe.")
            wheels = sorted((HERE / "wheels").glob("natten-*.whl"))
            if not wheels:
                raise SystemExit("No wheel in wheels/ to force-install.")
            pip_install(worker_py, ["--no-deps", "--force-reinstall", str(wheels[-1])])
        else:
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
