# Building NATTEN 0.21.0 on Windows from source

We ship a precompiled wheel in `wheels/`:

> `natten-0.21.0+winsm89-cp312-cp312-win_amd64.whl`

The wheel targets **`sm_89`** (Ada Lovelace) with PTX forward-compat, so it
runs natively on RTX 40-series cards and JIT-compiles on first launch for
sm_120 (RTX 5090) and sm_86 (RTX 30) hardware. If you need a wheel targeting
your own GPU SM exactly (or for a different Python / CUDA combo), follow these
steps.

## Prerequisites

| Tool | Version | How to install |
|---|---|---|
| Microsoft Visual Studio 2022 Build Tools | 14.40+ (MSVC v143) | `winget install Microsoft.VisualStudio.2022.BuildTools` then add the **"C++ build tools"** workload. |
| CUDA Toolkit | 12.8 (matching your worker pytorch cu128) | https://developer.nvidia.com/cuda-toolkit-archive |
| CMake | ≥ 3.27 | `pip install cmake` into the worker env (binary lands on `Scripts/`). |
| Ninja | any | `pip install ninja`. |
| Python | 3.12 (worker env) | comes with comfy-env / pixi. |
| Git | any | https://git-scm.com/ |

## Step-by-step

1. Clone NATTEN 0.21.0 + cutlass submodule:

   ```bat
   git clone --depth 1 --branch v0.21.0 https://github.com/SHI-Labs/NATTEN.git C:\Temp\NATTEN
   cd C:\Temp\NATTEN
   git submodule update --init --recursive --depth 1
   ```

2. Apply our Windows patches: copy these patched files in over the upstream
   ones (the diffs are in `patches/natten_msvc.diff` — apply manually or use
   `git apply`):

   - `csrc/CMakeLists.txt`     — wraps GCC-only flags in `if(NOT MSVC)`, adds
     `/Zc:__cplusplus /permissive- /bigobj` for cl.exe, filters hopper +
     blackwell `.cu` files out of `TORCH_APIS` when their gating macros
     aren't set.
   - `csrc/natten.cpp`         — wraps blackwell + hopper `m.def()` blocks
     in `#ifdef NATTEN_WITH_*` with runtime-throwing stub `else` branches
     (so `natten._libnatten` Python import still succeeds without those
     kernels).
   - `csrc/include/natten/helpers.h` — replaces `not x.is_sparse()` with
     `!x.is_sparse()` (nvcc parser rejects the C++ alt-token `not`).
   - `setup.py`                — emits both `XX-real` (SASS) and `XX-virtual`
     (PTX) entries in `CUDA_ARCHITECTURES`, so the wheel forward-compats to
     newer architectures (e.g. sm_120 on RTX 5090 from an sm_89 build).

3. Open a PowerShell as your user (no admin needed) and source vcvars:

   ```powershell
   $vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
   $envDump = cmd /c "`"$vcvars`" >nul 2>&1 && set"
   foreach ($line in $envDump) {
     if ($line -match '^([^=]+)=(.*)$') {
       [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
     }
   }
   $env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
   $env:CUDA_PATH = $env:CUDA_HOME
   $env:Path = "$env:CUDA_HOME\bin;$env:Path"
   $env:NATTEN_CUDA_ARCH = '8.9'   # set to your GPU's compute capability with a dot
   $env:NATTEN_WITH_CUDA = '1'
   $env:MAX_JOBS = '8'
   $env:DISTUTILS_USE_SDK = '1'
   ```

   Replace `'8.9'` with your card's compute capability written with a dot:

   | GPU | NATTEN_CUDA_ARCH |
   |---|---|
   | RTX 30 (Ampere) | `8.6` |
   | RTX 40 (Ada Lovelace) | `8.9` |
   | H100 / H200 (Hopper) | `9.0` |
   | RTX 50 (Blackwell consumer) | `8.9` *(no sm_120 kernels in 0.21.0 — sm_89 + PTX is the recommended fallback)* |
   | B100 / B200 (Blackwell data center) | `10.0` |

4. Build the wheel into a target directory:

   ```powershell
   Set-Location C:\Temp\NATTEN
   & '<worker_python>' -m pip wheel . `
     --no-build-isolation --no-deps `
     -w C:\Temp\natten_dist
   ```

   This takes 15-30 min depending on your CPU. Output is one `.whl` file in
   `C:\Temp\natten_dist`.

5. Install:

   ```powershell
   & '<worker_python>' -m pip install --no-deps --force-reinstall `
     C:\Temp\natten_dist\natten-0.21.0+*.whl
   ```

6. Verify:

   ```powershell
   & '<worker_python>' -c "import natten, torch; print(natten.__version__, natten.HAS_LIBNATTEN); print(natten.na2d(torch.randn(1,32,32,4,64,device='cuda',dtype=torch.float16), torch.randn(1,32,32,4,64,device='cuda',dtype=torch.float16), torch.randn(1,32,32,4,256,device='cuda',dtype=torch.float16), kernel_size=(9,9), backend='cutlass-fna').shape)"
   ```

   Expected output: `0.21.0 True` and `torch.Size([1, 32, 32, 4, 256])`.

## Why the patches are necessary

| Symptom from a vanilla build | Patch |
|---|---|
| `cl : Command line error D8021 : invalid numeric argument '/Wconversion'` | Drop GCC flags `-Wconversion`, `-fno-strict-aliasing`, `-ftemplate-backtrace-limit=0`, `-Wall`, `-O3` from MSVC build path. |
| `nvcc fatal : Unsupported gpu architecture 'compute_1200'` | natten's `_check_cuda_arch` multiplies float-parsed arch by 10. Pass `NATTEN_CUDA_ARCH=8.9` (with decimal) not `89`. |
| `Cannot open include file: 'cutlass/cutlass.h'` | `git submodule update --init --recursive --depth 1`. |
| `namespace "cutlass::platform" has no member "is_unsigned_v"` | cutlass header gates the alias on `__cplusplus >= 201703L`. MSVC needs `/Zc:__cplusplus` forwarded via `-Xcompiler`. |
| `error: identifier "not" is undefined` | nvcc's parser doesn't accept the C++ alt-token `not`. Replace with `!` in `csrc/include/natten/helpers.h`. |
| `warpspecialized.hpp(1550): error C2061: syntax error: identifier 'PipelineState'` | cutlass's Blackwell/Hopper warpspecialized templates don't compile under MSVC. Filter `*hopper_*.cu` and `*blackwell_*.cu` out of `TORCH_APIS` glob. |
| `LNK2019: unresolved external symbol "natten::blackwell_na1d_forward"` | `natten.cpp` registers pybind names unconditionally. Guard the blackwell/hopper `m.def()` blocks in `#ifdef` with runtime-throwing stub `else` branches. |
| `RuntimeError: no kernel image is available for execution on the device` on RTX 5090 | sm_89 SASS-only doesn't JIT to sm_120. Emit both `89-real` (SASS) **and** `89-virtual` (PTX) in `CUDA_ARCHITECTURES`. |
