# NeCA++ Environment Requirements

This guide covers the current `train.py` workflow in both 2D and 3D input modes,
including two-stage optimization, raw projection export, NIfTI export, and
3D-mode Dice evaluation. It is based on a source-code audit and the existing
`necacu13` Conda environment, inspected on 29 October 2026.

The accompanying `requirements.txt` is a curated list of project dependencies,
not a full export of every package installed in the development environment.
System tools and CERN TIGRE require the separate installation steps below.

## Reference platform

| Component | Observed reference installation |
|---|---|
| Platform | 64-bit Windows |
| Python | 3.10.19 |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA build | 12.8 |
| System CUDA Toolkit | 12.8; `nvcc` 12.8.61 |
| NVIDIA GPU | GeForce RTX 5080, approximately 16 GB VRAM |
| NVIDIA driver | 595.95 |
| C++ toolchain | Visual Studio 2022 Build Tools; installed MSVC toolset 14.44.35207 |
| Ninja | 1.13.1, installed through Conda |
| CERN TIGRE | Installed from the local `TIGRE-3.0/Python` source tree; distribution metadata reports `pytigre` 2.4.0 |

These are reference versions, not measured minimum requirements. GPU memory
usage depends on reconstruction resolution, the network, and `render.netchunk`.
The documented environment is the Windows reference setup; a fresh Linux
installation has not been validated as part of this audit.

## Hardware and build tools

- An NVIDIA CUDA-capable GPU and a driver compatible with the selected CUDA build.
- A full CUDA Toolkit with `nvcc`, headers, and development libraries. The CUDA
  runtime bundled with a PyTorch wheel alone is insufficient for compiling this
  project's hash-grid extension.
- On Windows, Visual Studio C++ Build Tools and a Windows SDK. Use an x64 MSVC
  toolchain supported by the installed CUDA Toolkit.
- Ninja, visible on `PATH` after activating the Conda environment.

The current implementation requires CUDA for both input modes. Selecting 2D
input only changes how observed projections are obtained; it still uses TIGRE
and the CUDA hash-grid encoder during reconstruction. It is not a CPU-only mode.

The hash-grid extension is compiled automatically when its Python backend is
first imported. Keep these project files available:

```text
src/encoder/hashencoder/src/hashencoder.cu
src/encoder/hashencoder/src/hashencoder.h
src/encoder/hashencoder/src/bindings.cpp
```

Use a writable PyTorch extension-cache location. The first compilation may take
longer than subsequent runs that reuse a compatible cached extension.

## Python dependencies

| Package | Reference version | Purpose |
|---|---|---|
| `torch` | 2.8.0+cu128 | Networks, optimization, autograd, and CUDA extension loading |
| `numpy` | 2.1.2 | Projection/volume arrays and percentile-based coarse masks |
| `scipy` | 1.15.2 | Three-dimensional morphological dilation |
| `PyYAML` | 6.0.3 | Training and geometry configuration files |
| `Pillow` | 11.3.0 | Raw projection PNG previews; imported as `PIL` |
| `nibabel` | 5.3.2 | NIfTI reconstruction and binary-mask export |
| `tqdm` | 4.65.0 | Training progress display |
| `tensorboardX` | 2.6.4 | Experiment logging through the existing fallback backend |
| `matplotlib` | 3.10.7 | Dependency imported by the installed TIGRE package |
| `Cython` | 3.2.4 | TIGRE extension build dependency and declared package dependency |
| `setuptools` | 75.8.2 | Source-package build tooling |
| `wheel` | 0.45.1 | Source-package wheel build tooling |

The reference environment uses `tensorboardX`: `torch.utils.tensorboard` is
attempted first, then the code falls back to `tensorboardX`. Installing the
separate `tensorboard` package is optional for this reference setup.

Let the package managers install required transitive dependencies, such as
`filelock`, `typing_extensions`, `sympy`, `networkx`, `jinja2`, `fsspec`,
`packaging`, `protobuf`, `importlib_resources`, and Windows `colorama`.
They are not unnecessary merely because project files do not import them directly.

## Installation in a separate environment

These commands document a new environment; they are not instructions to remove
packages from the existing working `necacu13` environment. Replace machine-specific
tool and TIGRE source paths when installing on another computer.

### 1. Prepare CUDA, MSVC, and Conda

Install the CUDA Toolkit and Visual Studio C++ Build Tools first. In an Anaconda
Prompt using CMD syntax, initialize the MSVC tools if they are not already available:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
conda create -n necapp python=3.10.19 pip=25.2 -y
conda activate necapp
conda install -c conda-forge ninja=1.13.1 -y

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"
```

Confirm the tools are visible in the same activated terminal:

```bat
where python
where nvcc
where ninja
where cl
nvcc --version
ninja --version
```

Keep PyTorch's CUDA build, the system Toolkit, and the host compiler compatible.
The existing Conda environment also contains CUDA 12.9 packages associated with
other installed software; this project was observed using system `nvcc` 12.8 and
PyTorch `cu128`. Do not infer the required Toolkit from a full Conda export alone.

### 2. Install PyTorch and the curated Python requirements

From the NeCA++ project root, install the CUDA 12.8 PyTorch build first:

```bat
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The PyTorch version and CUDA index follow the official
[PyTorch 2.8.0 installation options](https://pytorch.org/get-started/previous-versions/#v280).
`torchvision` and `torchaudio` are not used by the current project and are omitted.
The requirement `torch==2.8.0+cu128` is satisfied by the first installation command;
the PyTorch wheel index is not needed for the remaining packages.

### 3. Install CERN TIGRE from source

Use the same TIGRE source tree as the reference environment when reproducing it.
The recorded installation source is `C:\tigre\TIGRE-3.0\Python`. With that source
tree available, run:

```bat
python -m pip install --no-build-isolation "C:\tigre\TIGRE-3.0\Python"
```

TIGRE requires its parent source tree, including the sibling `Common` CUDA/C++
sources. Do not copy only its `Python` directory. Its build tools and Python
dependencies are installed by the previous steps.

The Python import is `tigre`, while installed distribution metadata is named
`pytigre`. In the inspected source tree, `setup.py` reports version `2.4.0` despite
the source directory being named `TIGRE-3.0`. A distribution version alone is
therefore not an exact source-revision identifier. Retain the actual source
archive or commit used for a release; this audit did not establish an upstream
commit for that local tree.

Use the [official CERN TIGRE repository](https://github.com/CERN/TIGRE) and its
[Python installation instructions](https://github.com/CERN/TIGRE/blob/v3.0/Frontispiece/python_installation.md)
when obtaining or rebuilding the source. A plain `pip install pytigre` is not
documented here as an equivalent replacement for the installed local source build.

## Verify the environment

Run the following commands after activating the environment and setting up the
compiler and CUDA paths:

```bat
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU detected')"
python -c "import numpy, scipy.ndimage, yaml, PIL, nibabel, tqdm, tensorboardX, tigre; print('Required imports: OK')"
python -c "from torch.utils.cpp_extension import CUDA_HOME, is_ninja_available; print('CUDA_HOME:', CUDA_HOME); print('Ninja available:', is_ninja_available())"
python -c "import tigre.utilities.gpu as gpu; print('TIGRE GPUs:', gpu.getGpuNames())"
```

Then, from the project root:

```bat
python train.py --help
```

Project imports occur before argument parsing, so this last command may compile
or load the CUDA hash-grid extension even though it only requests help.
Package imports and GPU detection alone do not validate a fresh source build or
a complete reconstruction; perform a small reconstruction in the new environment
before using it for experiments.

## Packages not required by the current project workflow

The current source does not require `astra-toolbox`, `odl`, `torchvision`,
`pandas`, or `openpyxl`. They were present in `necacu13` but are excluded from
the curated requirements. `matplotlib` is retained because TIGRE imports its
visualization modules, even though NeCA++ itself no longer plots with it.

This is a dependency specification for a fresh project environment, not a blanket
uninstall list for a shared environment. No installed packages were changed
during this audit.

## Scope of verification

The existing environment's package versions, CUDA availability, GPU identity,
Toolkit/compiler installation, required Python imports, logging fallback, and
Ninja availability on the activated Conda path were inspected. The installed
TIGRE source metadata and build requirements were also checked.

The installation steps above were not executed in a newly created environment,
and no complete GPU reconstruction was rerun for this documentation task.
