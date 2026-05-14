"""Prestartup: extend `comfy_env` worker timeout for Pixal3D nodes.

`comfy_env.isolation.metadata` hard-codes a 600s per-call timeout for proxied
node methods. Pixal3D's `pipeline.run()` (cold-load of 24 GB of weights +
4× DINOv3 + MoGe + NAF + the actual SS/Shape/Tex cascade) can exceed that on
the first invocation. We bump it to 3600s for any class named `Pixal3D*`.

Loaded BEFORE register_nodes() so the proxies pick up the patched timeout.

Defensive — not strictly required in the in-process registration model used by
this plugin's `nodes/__init__.py` (Pixal3D node classes are imported directly
into the main ComfyUI process, so they never go through `SubprocessWorker`).
The patch survives as cheap insurance against two paths:
  (a) future versions of this plugin that wrap Pixal3D in comfy_env after all,
  (b) downstream nodes (or `Pixal3D: Load Pipeline`-style warmup nodes embedded
      in TRELLIS2) that still proxy through the subprocess worker.
If `comfy_env` is not installed in the ComfyUI environment, the try/except
below makes this prestartup a no-op — Pixal3D itself doesn't depend on it.
"""

try:
    import comfy_env.isolation.workers.subprocess as _cews_sp

    # Idempotency sentinel: if our wrapper is already installed (e.g. comfy_env
    # reload or hot-reload tooling re-runs prestartup in the same process),
    # skip — otherwise we'd wrap the wrapper recursively and blow the stack
    # on the first node call.
    if getattr(_cews_sp.SubprocessWorker.call_method, "_pixal3d_patched", False):
        print("[ComfyUI-Pixal3D-prestartup] call_method already patched — skipping.")
    else:
        _orig_call_method = _cews_sp.SubprocessWorker.call_method

        def _pixal3d_patched_call(self, module_name, class_name, method_name,
                                  self_state=None, kwargs=None, timeout=None):
            if class_name and class_name.startswith("Pixal3D"):
                timeout = max(timeout or 0.0, 3600.0)
            return _orig_call_method(
                self,
                module_name=module_name,
                class_name=class_name,
                method_name=method_name,
                self_state=self_state,
                kwargs=kwargs,
                timeout=timeout,
            )

        _pixal3d_patched_call._pixal3d_patched = True
        _cews_sp.SubprocessWorker.call_method = _pixal3d_patched_call
        print("[ComfyUI-Pixal3D-prestartup] SubprocessWorker.call_method patched "
              "(timeout 3600s for Pixal3D*)")
except Exception as e:
    print(f"[ComfyUI-Pixal3D-prestartup] timeout patch skipped: {e}")
