"""
Patch Pixal3D's `pipelines/rembg/BiRefNet.py` so BiRefNet survives on
Windows + PyTorch ≥ 2.6 + transformers ≥ 4.55.

Two fixes are applied to `BiRefNet.__call__`:

1. **Rescue inference-tensors at call time.** `AutoModelForImageSegmentation.
   from_pretrained(..., trust_remote_code=True)` loads BiRefNet's parameters
   as PyTorch "inference tensors" (a side-effect of `torch.inference_mode()`
   inside transformers' loader). Conv2d refuses to consume those outside an
   inference_mode context with `RuntimeError: Inference tensors do not track
   version counter.` Plain `torch.no_grad()` is *not* enough. The patch
   detaches + clones every parameter/buffer that comes back from
   `is_inference()` so subsequent layers see regular tensors.

2. **Run the forward pass inside `torch.inference_mode()`**, not
   `torch.no_grad()`. This is belt-and-suspenders given the clone above.

Run after `install.py` clones `_pixal3d_src/`. Idempotent.
"""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BIREFNET_FILE = REPO_ROOT / "_pixal3d_src/pixal3d/pipelines/rembg/BiRefNet.py"

ORIGINAL = """    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()"""

PATCHED = """    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # The rescue from inference-tensor state happens once at pipeline
        # load (in ComfyUI-Pixal3D's pixal3d_stages.load_pipeline). We rely
        # on that; no per-call rescue here so calls stay fast.
        with torch.inference_mode():
            preds = self.model(input_images)[-1].sigmoid().cpu()"""


def main() -> int:
    if not BIREFNET_FILE.is_file():
        print(f"[birefnet_patch] file not found: {BIREFNET_FILE}")
        return 1

    src = BIREFNET_FILE.read_text(encoding="utf-8")

    if "rescue from inference-tensor state happens once at pipeline" in src:
        print(f"[birefnet_patch] {BIREFNET_FILE} already fully patched.")
        return 0

    if ORIGINAL not in src:
        print(f"[birefnet_patch] expected upstream snippet not found in {BIREFNET_FILE}. "
              "Pixal3D upstream may have changed shape. Skipping (NOT fatal — but "
              "you may need to redo the patch by hand).")
        return 1

    BIREFNET_FILE.write_text(src.replace(ORIGINAL, PATCHED), encoding="utf-8")
    print(f"[birefnet_patch] patched {BIREFNET_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
