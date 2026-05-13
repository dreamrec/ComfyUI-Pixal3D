"""
Patch valeoai/NAF's `src/layers/attentions.py` after `torch.hub.load` fetches it.

NAF imports natten's old split `na2d_qk` + `na2d_av` API. With natten 0.21+
those names are gone — only the unified `na2d(q, k, v, ...)` remains. NAF
already has a fallback path for the modern API (`NATTEN_RECENT = True`) but
it hard-codes `backend="cutlass-fna"`. That works on Linux + libnatten but
isn't strictly required on Windows; we keep it (our locally compiled natten
wheel ships libnatten so cutlass-fna is the fast path).

This script is a no-op if the cached NAF doesn't exist yet (i.e. before the
first Pixal3D run). It can be re-run safely.
"""

import pathlib
import sys

NAF_FILE = pathlib.Path.home() / ".cache/torch/hub/valeoai_NAF_main/src/layers/attentions.py"


def main():
    if not NAF_FILE.is_file():
        print(f"[naf_patch] no NAF cache yet at {NAF_FILE} — skipping "
              "(it will be downloaded on first pipeline load and is already "
              "compatible with this plugin's expected natten 0.21.x API).")
        return 0

    src = NAF_FILE.read_text(encoding="utf-8")
    # Sanity check — current upstream NAF already supports `na2d` modern API.
    if 'backend="cutlass-fna"' in src or "backend='cutlass-fna'" in src:
        print(f"[naf_patch] {NAF_FILE} already uses cutlass-fna backend, "
              "no edit needed.")
        return 0
    print(f"[naf_patch] {NAF_FILE} doesn't match expected upstream — leaving alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
