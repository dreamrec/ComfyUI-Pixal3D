"""
ComfyUI-Pixal3D installer.

Steps (idempotent — safe to re-run):
  1. Pick the target Python:
     • Legacy worker mode: if pozzettiandrea's ComfyUI-TRELLIS2 sibling is
       installed, install deps into its comfy-env worker python.
     • Standalone mode: otherwise install into the current `sys.executable`
       (the .venv ComfyUI Desktop / Manager runs install.py with).
  2. Clone TencentARC/Pixal3D@master into `_pixal3d_src/` (skip if present).
  3. `pip install --no-deps` the contents of `requirements.txt`.
  4. `pip install --no-deps` the bundled natten wheel from `wheels/`.
  5. Apply the BiRefNet inference-mode patch.
  6. Sanity-import pixal3d + natten + moge + utils3d in the target env.

Run from inside `custom_nodes/ComfyUI-Pixal3D/`:

    "<python>" install.py

Use `--python <path>` to override target detection.
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
        f"Could not find ComfyUI-TRELLIS2 next to {HERE}. "
        "Install https://github.com/pozzettiandrea/ComfyUI-TRELLIS2 first."
    )


_WORKER_PY_SUFFIXES = (
    # comfy-env / pixi layouts. `env/` is usually a symlink to
    # `.pixi/envs/default/`, but on some installs only one exists.
    pathlib.Path("env") / "python.exe",
    pathlib.Path(".pixi") / "envs" / "default" / "python.exe",
    # In-plugin Windows venv (e.g. ComfyUI-Env-Manager style).
    pathlib.Path("Scripts") / "python.exe",
)


def _candidate_pythons(roots: list[pathlib.Path]) -> list[pathlib.Path]:
    """Expand a list of env-root directories into possible worker python paths."""
    hits: list[pathlib.Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for sub in _WORKER_PY_SUFFIXES:
            p = root / sub
            if p.is_file():
                hits.append(p)
    return hits


def find_worker_python(trellis2_root: pathlib.Path) -> pathlib.Path:
    """Look for the worker python that comfy-env / comfy-env-manager created for TRELLIS2.

    Tries several layouts in order:
      1. An explicit `env_dir` recorded in `comfy-env-root.toml` (legacy).
      2. Any `_env_*` under `C:\\ce\\` (the original comfy-env install root —
         this is where TRELLIS2 actually runs, so it must win over any stub
         `Scripts/python.exe` left in the plugin dir by side experiments).
      3. Any `_env_*` directory living *inside* the TRELLIS2 plugin (newer
         comfy-env-manager layout puts the env there).

    Each candidate root is probed against `env/python.exe`,
    `.pixi/envs/default/python.exe`, and `Scripts/python.exe`.
    """
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover — pyproject pins py3.12
        tomllib = None  # type: ignore[assignment]

    # 1. Explicit env_dir in comfy-env-root.toml (legacy — newer comfy-env
    #    versions write [cuda]/[node_reqs] only, with no env path).
    if tomllib is not None:
        for marker in ("comfy-env-root.toml", ".comfy-env-root.toml"):
            p = trellis2_root / marker
            if not p.is_file():
                continue
            try:
                cfg = tomllib.loads(p.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError:
                continue
            # `env_dir` may live at the top level or under a section.
            explicit_roots: list[pathlib.Path] = []
            stack: list = [cfg]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    for k, v in node.items():
                        if k in ("env_dir", "env_root", "env_path") and isinstance(v, str):
                            explicit_roots.append(pathlib.Path(v))
                        elif isinstance(v, (dict, list)):
                            stack.append(v)
                elif isinstance(node, list):
                    stack.extend(node)
            for py in _candidate_pythons(explicit_roots):
                return py

    # 2. Original C:\ce install root — this is the comfy-env worker env that
    #    actually runs TRELLIS2 nodes, so prefer it over any in-plugin stub.
    fallback_root = pathlib.Path(r"C:\ce")
    if fallback_root.is_dir():
        # Sort newest-name-last so the most recent env wins.
        roots = sorted(fallback_root.glob("_env_*"), key=lambda p: p.name)
        for py in _candidate_pythons(list(reversed(roots))):
            return py

    # 3. In-plugin envs (newer comfy-env-manager drops `_env_*/` next to the plugin).
    in_plugin = sorted(trellis2_root.glob("_env_*"), key=lambda p: p.name)
    for py in _candidate_pythons(in_plugin):
        return py

    raise SystemExit(
        f"Could not locate the ComfyUI-Trellis2 worker python.\n"
        f"Looked under:\n"
        f"  • env_dir from {trellis2_root / 'comfy-env-root.toml'}\n"
        f"  • {trellis2_root}/_env_*/(env|.pixi/envs/default|Scripts)/python.exe\n"
        f"  • {fallback_root}/_env_*/(env|.pixi/envs/default|Scripts)/python.exe\n"
        f"Pass --python <path-to-worker-python.exe> to override, or make sure "
        f"ComfyUI-Trellis2 has been installed and its env initialised at least once."
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

    # Pip auto-enables `--require-hashes` for the WHOLE file as soon as ANY
    # line carries `--hash=`. VCS installs (git+https://...) can't be hashed,
    # so passing requirements.txt directly fails with:
    #   "Hash is required when a hash is supplied for another requirement"
    # Split into three buckets and install them with separate `pip install`
    # calls; each call enforces hash-mode only when its bucket has hashes.
    #
    # Bucket 1: VCS installs (no hashing possible) — install one at a time.
    # Bucket 2: hash-pinned wheels (require-hashes activates automatically).
    # Bucket 3: plain version specifiers.
    vcs: list[str] = []
    hashed_groups: list[list[str]] = []  # one group per `\` continuation block
    plain: list[str] = []

    current_hashed: list[str] = []
    for raw_line in req.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip trailing line-continuation markers but keep the requirement
        # text for the hash-bucket where it must travel together.
        is_continuation_target = raw_line.rstrip().endswith("\\")
        clean = line.rstrip("\\").strip()
        if not clean:
            continue
        if clean.startswith("--hash="):
            current_hashed.append(clean)
            if not is_continuation_target:
                hashed_groups.append(current_hashed)
                current_hashed = []
            continue
        if is_continuation_target:
            # Requirement that opens a hashed block.
            current_hashed = [clean]
            continue
        if clean.startswith(("git+", "hg+", "svn+", "bzr+")):
            vcs.append(clean)
        else:
            plain.append(clean)
    # Anything left in current_hashed is malformed continuation; promote it
    # to the hashed bucket so pip surfaces a clear error instead of silently
    # dropping it.
    if current_hashed:
        hashed_groups.append(current_hashed)

    # --no-deps to avoid pip resolver fighting with the pixi-managed env.
    for vcs_url in vcs:
        pip_install(python, ["--no-deps", vcs_url])
    for group in hashed_groups:
        pip_install(python, ["--no-deps", *group])
    if plain:
        pip_install(python, ["--no-deps", *plain])


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

    # Two install modes:
    #   • Worker mode (legacy): pozzettiandrea's ComfyUI-TRELLIS2 sibling is
    #     present and its comfy-env subprocess runs Pixal3D nodes — deps go
    #     into that worker python.
    #   • Standalone mode: no TRELLIS2 sibling. Pixal3D nodes run in-process
    #     in ComfyUI's main Python; deps go into sys.executable (which the
    #     Comfy Registry / Manager invoke install.py with).
    # We try worker mode first for backward compat; if no TRELLIS2 is found,
    # gracefully fall back to standalone instead of hard-failing.
    if args.python:
        worker_py = pathlib.Path(args.python)
        log(f"Target python (explicit): {worker_py}")
    else:
        try:
            trellis2 = find_trellis2_root()
            log(f"ComfyUI-Trellis2 at: {trellis2}")
            worker_py = find_worker_python(trellis2)
            log(f"Worker python: {worker_py}")
        except SystemExit as e:
            worker_py = pathlib.Path(sys.executable)
            log(f"No ComfyUI-TRELLIS2 sibling — standalone mode.")
            log(f"Target python: {worker_py}")

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
