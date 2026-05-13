"""
Pixal3D pipeline stages — cached loaders + single-call run wrapper.

Lives alongside TRELLIS2's stages.py and shares the same in-process import
pattern: the comfy_env plumbing on ComfyUI-TRELLIS2's __init__.py already
exposes the isolated venv (o_voxel, cumesh, flex_gemm, flash_attn, etc.) on
sys.path, so we just need to add the cloned Pixal3D source to it.
"""

import gc
import logging
import math
import os
import pathlib
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

import comfy.model_management as mm

log = logging.getLogger("trellis2.pixal3d")

# ---------------------------------------------------------------------------
# sys.path bootstrap
# ---------------------------------------------------------------------------

# The Pixal3D python package is checked out under
# `custom_nodes/ComfyUI-Pixal3D/_pixal3d_src/` by `install.py`. Prepend it so
# `import pixal3d` resolves.
_CUSTOM_NODE_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PIXAL3D_SRC = _CUSTOM_NODE_ROOT / "_pixal3d_src"
if _PIXAL3D_SRC.is_dir() and str(_PIXAL3D_SRC) not in sys.path:
    sys.path.insert(0, str(_PIXAL3D_SRC))

# Force the sane attention backend — inference.py hardcodes flash_attn_3 (HF
# Spaces only). On Windows we have flash_attn 2.8.3 from the isolated venv.
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(_CUSTOM_NODE_ROOT / "pixal3d_autotune_cache.json"),
)

# Override any HF mirror that may leak in from the comfy_env worker env.
# Observed in the wild: HF_ENDPOINT inherited as https://hf-mirror.com from
# ComfyUI Desktop's launch env, which blocks BiRefNet/RMBG-2.0 download.
# We always want huggingface.co for this codepath.
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

# ---------------------------------------------------------------------------
# Constants (mirror inference.py)
# ---------------------------------------------------------------------------

MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
PIXAL3D_MODEL_PATH = "TencentARC/Pixal3D"
DINOV3_REPO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": DINOV3_REPO,
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": DINOV3_REPO,
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": DINOV3_REPO,
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": DINOV3_REPO,
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_pipeline = None        # Pixal3DImageTo3DPipeline (with image_cond_models attached)
_moge_model = None      # MoGeModel

PIPELINE_TYPES = ("1024_cascade", "1536_cascade")


def _rescue_inference_tensors(module) -> None:
    """Detach + clone every parameter and buffer of an nn.Module so they
    become regular (non-inference) tensors.

    ComfyUI's `execution.py:734` wraps every node FUNCTION call in
    `torch.inference_mode()`. Inside that ambient context, **any** new
    tensor is an inference tensor — including `.detach().clone()` results.
    To produce regular tensors we have to **explicitly disable**
    inference_mode for the duration of the clone, via
    `torch.inference_mode(mode=False)`. Without this guard the rescue is
    silently a no-op when called from a ComfyUI node.

    Safe to call repeatedly (idempotent).
    """
    with torch.inference_mode(mode=False), torch.no_grad():
        sd = {k: v.detach().clone() for k, v in module.state_dict().items()}
        module.load_state_dict(sd, strict=True)
        for _p in module.parameters():
            _p.data = _p.data.detach().clone()
            _p.requires_grad_(False)
        for _b in module.buffers():
            _b.data = _b.data.detach().clone()


def _build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )
    m = DinoV3ProjFeatureExtractor(**config)
    m.eval()
    # DinoV3ProjFeatureExtractor wraps a transformers DINOv3ViTModel that
    # comes back with inference-tensor weights. Rescue them now so the first
    # forward pass doesn't fail at patch_embeddings (Conv2d).
    if hasattr(m, "model"):
        _rescue_inference_tensors(m.model)
    return m


def load_pipeline(low_vram: bool = False, model_path: str = PIXAL3D_MODEL_PATH):
    """Load and cache the Pixal3D pipeline with all 4 DinoV3 projection extractors.

    Mirrors inference.py:init_pipeline() but caches at module scope so the next
    queue reuses everything.
    """
    global _pipeline
    if _pipeline is not None:
        # Toggle low_vram dynamically if user changed it.
        _pipeline.low_vram = low_vram
        return _pipeline

    # Belt-and-suspenders override: HF_ENDPOINT may be inherited from
    # ComfyUI's launch env as https://hf-mirror.com (observed in the worker
    # log). The env var override only helps if huggingface_hub hasn't been
    # imported yet; patch its already-cached constants in case it has.
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    try:
        import huggingface_hub.constants as _hfc
        _hfc.ENDPOINT = "https://huggingface.co"
        _hfc._HF_DEFAULT_ENDPOINT = "https://huggingface.co"
    except Exception:
        pass
    try:
        import transformers.utils.hub as _th
        _th.HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"
    except Exception:
        pass

    from pixal3d.pipelines import Pixal3DImageTo3DPipeline
    from pixal3d.pipelines.rembg.BiRefNet import BiRefNet

    log.info(f"[Pixal3D] Loading pipeline from {model_path}...")
    # Override the gated briaai/RMBG-2.0 rembg model BEFORE from_pretrained
    # runs by monkey-patching the pipeline.json args. The pipeline.json is in
    # the HF cache snapshot; we override at the rembg class level instead by
    # subclassing — simpler than rewriting JSON in cache.
    _orig_init = BiRefNet.__init__
    def _patched_init(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # If caller passed the gated repo (default in upstream pipeline.json),
        # transparently swap to the ungated mirror.
        if model_name in ("briaai/RMBG-2.0", "briaai/RMBG-1.4"):
            log.info(f"[Pixal3D] Rerouting rembg {model_name} -> ZhengPeng7/BiRefNet (ungated)")
            model_name = "ZhengPeng7/BiRefNet"
        _orig_init(self, model_name=model_name)
    BiRefNet.__init__ = _patched_init

    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    # transformers' from_pretrained loads weights as inference-tensors which
    # Conv2d refuses to use outside inference_mode. Rescue rembg + the DINOv3
    # extractors (via _build_image_cond_model) once at load time — that's
    # enough; inference-tensor state doesn't drift across queue runs.
    if getattr(pipeline, "rembg_model", None) is not None and hasattr(pipeline.rembg_model, "model"):
        _rescue_inference_tensors(pipeline.rembg_model.model)
        pipeline.rembg_model.model.eval()
        log.info("[Pixal3D] Rescued rembg_model from inference-tensor state")

    log.info("[Pixal3D] Building 4x DinoV3ProjFeatureExtractor...")
    pipeline.image_cond_model_ss = _build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = _build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = _build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = _build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    pipeline.low_vram = low_vram

    device = mm.get_torch_device()
    pipeline.cuda() if device.type == "cuda" else pipeline.to(device)
    for attr in (
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    ):
        m = getattr(pipeline, attr, None)
        if m is None:
            continue
        if device.type == "cuda":
            m.cuda()
        else:
            m.to(device)

    log.info("[Pixal3D] Pre-loading NAF upsamplers...")
    for attr in (
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    ):
        m = getattr(pipeline, attr, None)
        if m is not None and getattr(m, "use_naf_upsample", False):
            m._load_naf()

    _pipeline = pipeline
    log.info("[Pixal3D] Pipeline ready")
    return _pipeline


def load_moge():
    """Load and cache MoGe-2 for camera intrinsics estimation."""
    global _moge_model
    if _moge_model is not None:
        return _moge_model

    from moge.model.v2 import MoGeModel

    log.info(f"[Pixal3D] Loading MoGe-2 from {MOGE_MODEL_NAME}...")
    device = mm.get_torch_device()
    m = MoGeModel.from_pretrained(MOGE_MODEL_NAME).to(device)
    m.eval()
    # MoGe's huggingface_hub loader returns inference-tensor weights, which
    # then fail in torch.cat with "Inference tensors do not track version
    # counter." Rescue all params + buffers into regular tensors.
    _rescue_inference_tensors(m)
    _moge_model = m
    return _moge_model


# ---------------------------------------------------------------------------
# Camera estimation (verbatim from inference.py, packaged as a function)
# ---------------------------------------------------------------------------


def _compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
    return float(focal_length * resolution / 32.0)


def _distance_from_fov(
    camera_angle_x: float,
    grid_point: torch.Tensor,
    target_point: torch.Tensor,
    mesh_scale: float,
    image_resolution: int,
) -> dict:
    rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    gp = grid_point.to(torch.float32) @ rotation.T
    gp = gp / mesh_scale / 2
    xw, yw = gp[0].item(), gp[1].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = _compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def estimate_camera(
    pil_image: Image.Image,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
) -> dict:
    """Estimate camera intrinsics + distance from a PIL image (already preprocessed)."""
    moge = load_moge()
    device = mm.get_torch_device()

    width, height = pil_image.size
    img_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).to(device)

    with torch.inference_mode():
        out = moge.infer(img_t)
    intrinsics = out["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    target = torch.tensor(
        [0 - extend_pixel, image_resolution - 1 + extend_pixel],
        dtype=torch.float32,
    )
    distance = _distance_from_fov(
        camera_angle_x, grid_point, target, mesh_scale, image_resolution
    )["distance_from_x"]
    return {
        "camera_angle_x": camera_angle_x,
        "distance": distance,
        "mesh_scale": mesh_scale,
    }


# ---------------------------------------------------------------------------
# ComfyUI <-> PIL bridging
# ---------------------------------------------------------------------------


def _comfy_image_to_pil(image: torch.Tensor, mask: Optional[torch.Tensor]) -> Image.Image:
    """ComfyUI IMAGE [B,H,W,C] (+ optional MASK [B,H,W]) -> PIL RGB or RGBA.

    If mask is supplied, attach as alpha so pipeline.preprocess_image skips
    BiRefNet and goes straight to smart-crop + composite.
    """
    if image.dim() == 4:
        rgb = image[0]
    else:
        rgb = image
    rgb_np = (rgb.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(rgb_np, "RGB")

    if mask is None:
        return pil

    mn = mask.detach().cpu().numpy()
    if mn.ndim == 4:
        mn = mn[0]
    if mn.ndim == 3:
        mn = mn[..., 0] if mn.shape[-1] in (1, 2, 3, 4) else mn[0]
    if mn.ndim != 2:
        log.warning(f"[Pixal3D] Mask shape {mn.shape} not 2D; ignoring")
        return pil

    if mn.shape != (pil.height, pil.width):
        mask_pil = Image.fromarray((mn * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((pil.width, pil.height), Image.LANCZOS)
        mn = np.array(mask_pil) / 255.0

    alpha = (mn * 255).astype(np.uint8)
    rgba_np = np.dstack([np.array(pil), alpha])
    return Image.fromarray(rgba_np, "RGBA")


def _pil_to_comfy_image(pil: Image.Image) -> torch.Tensor:
    arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


_BG_COLOR_MAP = {
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "white": (255, 255, 255),
}


def run_pixal3d(
    image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    low_vram: bool = False,
    seed: int = 42,
    pipeline_type: str = "1024_cascade",
    max_num_tokens: int = 49152,
    mesh_scale: float = 1.0,
    image_resolution: int = 512,
    background_color: str = "gray",
    camera_angle_x_override: float = 0.0,
    distance_override: float = 0.0,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_guidance_strength: float = 7.5,
    shape_guidance_rescale: float = 0.5,
    shape_sampling_steps: int = 12,
    shape_rescale_t: float = 3.0,
    tex_guidance_strength: float = 1.0,
    tex_guidance_rescale: float = 0.0,
    tex_sampling_steps: int = 12,
    tex_rescale_t: float = 3.0,
    decimation_target: int = 200000,
    texture_size: int = 2048,
    remesh: bool = True,
) -> Tuple[Any, torch.Tensor, dict]:
    """Full Pixal3D image-to-mesh pipeline.

    Returns: (trimesh_scene, preprocessed_comfy_image, camera_params_dict)
    """
    # ComfyUI's execution.py:734 wraps every node FUNCTION call in
    # `torch.inference_mode()`. Inside that ambient context, every tensor
    # created (including model weights loaded by transformers' from_pretrained)
    # becomes an "inference tensor" which Conv2d / matmul / torch.cat reject
    # with "Inference tensors do not track version counter." Explicitly
    # disable the ambient inference_mode for the whole run; use no_grad for
    # autograd discipline.
    with torch.inference_mode(mode=False), torch.no_grad():
        return _run_pixal3d_body(
            image=image, mask=mask, low_vram=low_vram, seed=seed,
            pipeline_type=pipeline_type, max_num_tokens=max_num_tokens,
            mesh_scale=mesh_scale, image_resolution=image_resolution,
            background_color=background_color,
            camera_angle_x_override=camera_angle_x_override,
            distance_override=distance_override,
            ss_guidance_strength=ss_guidance_strength,
            ss_guidance_rescale=ss_guidance_rescale,
            ss_sampling_steps=ss_sampling_steps, ss_rescale_t=ss_rescale_t,
            shape_guidance_strength=shape_guidance_strength,
            shape_guidance_rescale=shape_guidance_rescale,
            shape_sampling_steps=shape_sampling_steps,
            shape_rescale_t=shape_rescale_t,
            tex_guidance_strength=tex_guidance_strength,
            tex_guidance_rescale=tex_guidance_rescale,
            tex_sampling_steps=tex_sampling_steps,
            tex_rescale_t=tex_rescale_t,
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=remesh,
        )


def _run_pixal3d_body(*, image, mask, low_vram, seed, pipeline_type,
                      max_num_tokens, mesh_scale, image_resolution,
                      background_color, camera_angle_x_override,
                      distance_override, ss_guidance_strength,
                      ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
                      shape_guidance_strength, shape_guidance_rescale,
                      shape_sampling_steps, shape_rescale_t,
                      tex_guidance_strength, tex_guidance_rescale,
                      tex_sampling_steps, tex_rescale_t,
                      decimation_target, texture_size, remesh):
    import o_voxel

    if pipeline_type not in PIPELINE_TYPES:
        raise ValueError(f"pipeline_type must be in {PIPELINE_TYPES}, got {pipeline_type!r}")

    bg_color = _BG_COLOR_MAP.get(background_color, _BG_COLOR_MAP["gray"])
    log.info(f"[Pixal3D] background_color={background_color} -> {bg_color}")

    pipeline = load_pipeline(low_vram=low_vram)
    pil = _comfy_image_to_pil(image, mask)

    log.info("[Pixal3D] Preprocessing image (crop + composite)...")
    preprocessed = pipeline.preprocess_image(pil, bg_color=bg_color)

    # Camera params
    if camera_angle_x_override > 0.0 and distance_override > 0.0:
        camera_params = {
            "camera_angle_x": float(camera_angle_x_override),
            "distance": float(distance_override),
            "mesh_scale": float(mesh_scale),
        }
        log.info(f"[Pixal3D] Using manual camera: {camera_params}")
    else:
        log.info("[Pixal3D] Estimating camera with MoGe-2...")
        camera_params = estimate_camera(
            preprocessed,
            mesh_scale=mesh_scale,
            extend_pixel=0,
            image_resolution=image_resolution,
        )
        log.info(
            f"[Pixal3D] Camera: angle_x={camera_params['camera_angle_x']:.4f} rad, "
            f"distance={camera_params['distance']:.4f}"
        )

    torch.manual_seed(seed)

    ss_params = {
        "steps": ss_sampling_steps,
        "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale,
        "rescale_t": ss_rescale_t,
    }
    shape_params = {
        "steps": shape_sampling_steps,
        "guidance_strength": shape_guidance_strength,
        "guidance_rescale": shape_guidance_rescale,
        "rescale_t": shape_rescale_t,
    }
    tex_params = {
        "steps": tex_sampling_steps,
        "guidance_strength": tex_guidance_strength,
        "guidance_rescale": tex_guidance_rescale,
        "rescale_t": tex_rescale_t,
    }

    log.info(f"[Pixal3D] Running pipeline ({pipeline_type})...")
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=tex_params,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )

    # Free latents we don't keep around
    del shape_slat, tex_slat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mesh = mesh_list[0]
    log.info(f"[Pixal3D] Extracting GLB (decim={decimation_target}, tex={texture_size})...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )

    # Same rotation as inference.py (Pixal3D space -> conventional Y-up)
    rot = np.array(
        [
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    glb.apply_transform(rot)

    # Side-effect: drop a timestamped .glb in ComfyUI's output dir so users
    # always get a real artifact even without chaining an export node.
    try:
        import folder_paths, time as _t
        out_dir = pathlib.Path(folder_paths.get_output_directory())
        out_dir.mkdir(parents=True, exist_ok=True)
        glb_path = out_dir / f"pixal3d_{int(_t.time())}_{seed}.glb"
        # extension_webp=True (the upstream inference.py default) embeds WebP
        # textures which Blender/STB cannot decode. Stick to PNG for max compat.
        glb.export(str(glb_path))
        log.info(f"[Pixal3D] GLB written: {glb_path}")
    except Exception as e:
        log.warning(f"[Pixal3D] GLB auto-save failed: {e}")

    preprocessed_tensor = _pil_to_comfy_image(preprocessed)
    return glb, preprocessed_tensor, camera_params


def free_pipeline():
    """Drop cached pipeline + MoGe so the next run reloads (used by debug nodes)."""
    global _pipeline, _moge_model
    _pipeline = None
    _moge_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
