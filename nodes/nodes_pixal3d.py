"""ComfyUI nodes for Pixal3D (pixel-aligned image-to-3D)."""

import logging

log = logging.getLogger("trellis2.pixal3d")


PIPELINE_TYPES = ["1024_cascade", "1536_cascade"]


class Pixal3DLoadPipeline:
    """Force-load the Pixal3D pipeline (4x DinoV3 + SS/Shape/Tex flows + decoders) into VRAM.

    Optional. The image-to-mesh node loads on demand. Use this when you want
    the load time to show up in its own node for clarity, or to pre-warm models
    before a sweep.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low_vram": ("BOOLEAN", {"default": False, "tooltip": "Offload submodels to CPU between stages. Recommended for <24GB cards."}),
            }
        }

    RETURN_TYPES = ("PIXAL3D_PIPELINE_READY",)
    RETURN_NAMES = ("pipeline_ready",)
    FUNCTION = "load"
    CATEGORY = "Pixal3D"
    DESCRIPTION = "Pre-load Pixal3D pipeline + DinoV3 extractors + MoGe-2."

    def load(self, low_vram=False):
        from .pixal3d_stages import load_pipeline, load_moge

        load_pipeline(low_vram=low_vram)
        load_moge()
        return ({"low_vram": low_vram},)


class Pixal3DImageToMesh:
    """Single-image -> textured 3D mesh via Pixal3D pipeline.

    Mirrors inference.py: optional mask gets attached as alpha (skipping
    BiRefNet); otherwise the pipeline's own rembg runs. MoGe-2 estimates
    camera intrinsics unless camera_angle_x and distance are both > 0.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image (RGB). Single image; first batch entry is used."}),
                "pipeline_type": (PIPELINE_TYPES, {"default": "1024_cascade", "tooltip": "1024_cascade is recommended; 1536_cascade is heavier and may overflow VRAM."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Foreground mask (white=object). If provided, BiRefNet is skipped."}),
                "pipeline_ready": ("PIXAL3D_PIPELINE_READY",),
                "low_vram": ("BOOLEAN", {"default": False, "tooltip": "Only used if pipeline_ready is not connected."}),
                "max_num_tokens": ("INT", {"default": 49152, "min": 16384, "max": 131072, "step": 1024, "tooltip": "HR token budget. Lower = less VRAM, smaller mesh. 49152 ~= 9GB activations."}),
                "mesh_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "image_resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "camera_angle_x_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.14, "step": 0.001, "tooltip": "0 = auto (MoGe-2). Set both override fields > 0 to skip MoGe."}),
                "distance_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.01}),
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "ss_guidance_rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "ss_rescale_t": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "shape_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shape_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "shape_guidance_rescale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "shape_rescale_t": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "tex_guidance_strength": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "tex_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "tex_guidance_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "tex_rescale_t": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "decimation_target": ("INT", {"default": 200000, "min": 5000, "max": 1000000, "step": 5000}),
                "texture_size": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 256}),
                "remesh": ("BOOLEAN", {"default": True}),
                # New widgets MUST be appended below — older workflow JSONs
                # use positional widgets_values lists and shift if we insert
                # earlier. CI workflow at .github/workflows/ci.yml enforces
                # `len(widgets_values) >= len(INPUT_TYPES)` not strict equality.
                "background_color": (["gray", "black", "white"], {"default": "gray", "tooltip": "Color composited behind the foreground mask. Default 'gray' (128,128,128) prevents dark-silhouette bleed that bakes thin black lines into the PBR texture; 'black' matches Pixal3D inference.py defaults but causes those artifacts; 'white' for light-on-dark subjects."}),
                "keep_warm": ("BOOLEAN", {"default": True, "tooltip": "True: keep the loaded Pixal3D pipeline in VRAM after this run (~14 GB) so the NEXT call is fast (~3 min instead of ~7-10 min cold-load). False: auto-free the pipeline at the end of THIS call. Use False if you're done with Pixal3D for a while and need that VRAM back for other nodes."}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("mesh", "preprocessed_image", "camera_angle_x", "distance")
    FUNCTION = "generate"
    CATEGORY = "Pixal3D"
    DESCRIPTION = """Pixal3D pixel-aligned image-to-3D.

Outputs a textured trimesh (PBR) plus the preprocessed image and the
camera params actually used (helpful for retrying with manual overrides).
Chain TRIMESH into Trellis2RenderPreview / Trellis2ExportGLB."""

    def generate(
        self,
        image,
        pipeline_type="1024_cascade",
        seed=42,
        mask=None,
        pipeline_ready=None,
        low_vram=False,
        max_num_tokens=49152,
        mesh_scale=1.0,
        image_resolution=512,
        background_color="gray",
        camera_angle_x_override=0.0,
        distance_override=0.0,
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        ss_guidance_rescale=0.7,
        ss_rescale_t=5.0,
        shape_guidance_strength=7.5,
        shape_sampling_steps=12,
        shape_guidance_rescale=0.5,
        shape_rescale_t=3.0,
        tex_guidance_strength=1.0,
        tex_sampling_steps=12,
        tex_guidance_rescale=0.0,
        tex_rescale_t=3.0,
        decimation_target=200000,
        texture_size=2048,
        remesh=True,
        keep_warm=True,
    ):
        from .pixal3d_stages import run_pixal3d, free_pipeline

        effective_low_vram = pipeline_ready["low_vram"] if pipeline_ready else low_vram

        glb, preprocessed_tensor, cam = run_pixal3d(
            image,
            mask=mask,
            low_vram=effective_low_vram,
            seed=seed,
            pipeline_type=pipeline_type,
            max_num_tokens=max_num_tokens,
            mesh_scale=mesh_scale,
            image_resolution=image_resolution,
            background_color=background_color,
            camera_angle_x_override=camera_angle_x_override,
            distance_override=distance_override,
            ss_guidance_strength=ss_guidance_strength,
            ss_guidance_rescale=ss_guidance_rescale,
            ss_sampling_steps=ss_sampling_steps,
            ss_rescale_t=ss_rescale_t,
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

        # `keep_warm=False` is for users sharing the GPU with other
        # workflows — release the ~14 GB pipeline singleton now rather
        # than making them queue Pixal3D: Free Pipeline manually. The
        # next Pixal3D call will pay the full cold-load again.
        if not keep_warm:
            log.info("[Pixal3D] keep_warm=False — releasing pipeline singletons")
            free_pipeline()

        return (glb, preprocessed_tensor, cam["camera_angle_x"], cam["distance"])


class Pixal3DFreePipeline:
    """Drop cached Pixal3D pipeline + MoGe to free VRAM (no-op if nothing loaded)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "free"
    OUTPUT_NODE = True
    CATEGORY = "Pixal3D"
    DESCRIPTION = "Release Pixal3D / MoGe singletons."

    def free(self):
        from .pixal3d_stages import free_pipeline
        free_pipeline()
        return {}


NODE_CLASS_MAPPINGS = {
    "Pixal3DLoadPipeline": Pixal3DLoadPipeline,
    "Pixal3DImageToMesh": Pixal3DImageToMesh,
    "Pixal3DFreePipeline": Pixal3DFreePipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixal3DLoadPipeline": "Pixal3D: Load Pipeline",
    "Pixal3DImageToMesh": "Pixal3D: Image to Mesh",
    "Pixal3DFreePipeline": "Pixal3D: Free Pipeline",
}
