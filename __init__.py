"""ComfyUI-Pixal3D — Pixel-Aligned 3D Generation for ComfyUI.

This package depends on ComfyUI-Trellis2 (visualbruno/ComfyUI-Trellis2) being
installed as a sibling custom_node — it reuses TRELLIS.2's pixi-managed
worker env (24 GB CUDA + flash_attn stack) and its `comfy_env` worker plumbing.
"""

import logging
import pathlib
import sys

log = logging.getLogger("pixal3d")

# Add Pixal3D upstream source to sys.path. install.py git-clones it here.
_HERE = pathlib.Path(__file__).resolve().parent
_PIXAL3D_SRC = _HERE / "_pixal3d_src"
if _PIXAL3D_SRC.is_dir() and str(_PIXAL3D_SRC) not in sys.path:
    sys.path.insert(0, str(_PIXAL3D_SRC))

try:
    from .nodes.nodes_pixal3d import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception as e:
    log.warning(f"[ComfyUI-Pixal3D] node import failed: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
