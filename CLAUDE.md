# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IREE (Intermediate Representation Execution Environment) is an MLIR-based end-to-end compiler and runtime for machine learning models. It compiles ML models from frameworks like PyTorch, TensorFlow, and ONNX to optimized binaries that run on diverse hardware (CPU, GPU, accelerators).

**Architecture**: Framework → Input Dialect (StableHLO/TOSA/Torch) → IREE Compiler → VM Bytecode/HAL Executables → IREE Runtime

## Build System

IREE uses **CMake with Ninja** as the primary build system. Bazel build files exist but are automatically converted to CMake via `build_tools/bazel_to_cmake/bazel_to_cmake.py`.

### Initial Setup

```bash
# Clone with submodules
git clone https://github.com/iree-org/iree.git
cd iree
git submodule update --init

# Configure (macOS example)
cmake -G Ninja -B ../iree-build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON

# Build
cmake --build ../iree-build/
```

### Build Variants

- **Disable most backends** (faster builds): `-DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_LLVM_CPU=ON -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON`
- **Enable CUDA**: `-DIREE_TARGET_BACKEND_CUDA=ON -DIREE_HAL_DRIVER_CUDA=ON`
- **Enable ROCm/HIP**: `-DIREE_TARGET_BACKEND_ROCM=ON -DIREE_HAL_DRIVER_HIP=ON`
- **Python bindings**: `-DIREE_BUILD_PYTHON_BINDINGS=ON -DPython3_EXECUTABLE="$(which python3)"`

## Common Development Commands

### Building

```bash
# Build all targets
cmake --build ../iree-build/

# Build specific target
cmake --build ../iree-build/ --target iree-run-module

# Build test dependencies (required for many tests)
cmake --build ../iree-build/ --target iree-test-deps
```

### Testing

```bash
# Run all tests via CTest
ctest --test-dir ../iree-build/

# Run specific test (CMake)
ctest -R iree/base/bitfield_test

# Run with helper script (supports env vars for filtering)
export IREE_CUDA_DISABLE=0
export IREE_VULKAN_DISABLE=1
./build_tools/cmake/ctest_all.sh ../iree-build

# Build and run all tests in one command
cmake --build ../iree-build/ --target iree-run-tests
```

### Python Bindings

```bash
# Install build requirements
python -m pip install --upgrade pip
python -m pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt

# After building with Python bindings enabled:
CMAKE_INSTALL_METHOD=ABS_SYMLINK python -m pip install -e ../iree-build/compiler
CMAKE_INSTALL_METHOD=ABS_SYMLINK python -m pip install -e ../iree-build/runtime

# Or extend PYTHONPATH
source ../iree-build/.env && export PYTHONPATH
python -c "import iree.compiler; import iree.runtime"
```

## High-Level Architecture

### Compiler (`compiler/`)

**Location**: `compiler/src/iree/compiler/`

Key components:
- **API/**: Public C and Python APIs
- **Codegen/**: Device code generation (compute kernels)
- **ConstEval/**: JIT constant evaluation
- **Dialect/**: Core MLIR dialects
  - `Flow/`: Tensor program modeling, compute workload partitioning
  - `HAL/`: Hardware Abstraction Layer (buffer/execution management)
  - `Stream/`: Device placement, async scheduling
  - `VM/`: Virtual Machine dialect
  - `Util/`: Common types
- **InputConversion/**: Conversions from frontend/input dialects
- **Pipelines/**: Translation pipeline definitions
- **PluginAPI/**: Plugin system for extensibility

**Coding style**: LLVM/Clang-style (not Google style like runtime)

### Compiler Plugins (`compiler/plugins/`)

Extensible plugin architecture for adding:
- **Input dialects** (`plugins/input/`): StableHLO, TOSA, Torch
- **Target backends** (`plugins/target/`): LLVMCPU, CUDA, ROCm, VulkanSPIRV, MetalSPIRV, VMVX, WebGPU

Plugins are registered via `iree_compiler_plugin.cmake` and activated at session level.

### Runtime (`runtime/`)

**Location**: `runtime/src/iree/`

Key components:
- **base/**: Common types and utilities
- **hal/**: Hardware Abstraction Layer implementations
  - Drivers: local-sync, local-task, cuda, vulkan, hip, metal, null
  - Target backends map to specific drivers
- **vm/**: Virtual Machine (bytecode execution)
- **task/**: Multi-threaded task execution
- **schemas/**: FlatBuffers-based data formats
- **modules/**: Standard modules (UKernels, device)
- **io/**: I/O and parameter management
- **tooling/**: Test utilities (not for production use)

**Coding style**: Google style (different from compiler)

## Developer Tools

Located in `tools/`, built to `iree-build/tools/`:

- **iree-opt**: Test compiler passes (like mlir-opt but for IREE)
  ```bash
  ../iree-build/tools/iree-opt --iree-util-drop-compiler-hints test.mlir
  ../iree-build/tools/iree-opt --iree-transformation-pipeline --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx model.mlir
  ```

- **iree-compile**: Main compiler driver
  ```bash
  ../iree-build/tools/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvf mlir -o module.vmfb
  ```

- **iree-run-module**: Execute compiled modules
  ```bash
  ../iree-build/tools/iree-run-module --module=module.vmfb --device=local-task --function=main --input=f32=42
  ```

- **iree-check-module**: Run e2e check framework tests
  ```bash
  ../iree-build/tools/iree-check-module --device=local-task --module=test.vmfb
  ```

- **iree-run-mlir**: Compile and execute in one step (debugging)
- **iree-dump-module**: Inspect FlatBuffer module files
- **iree-tokenize**: Tokenize text for LLM inference

## Testing Framework

### Compiler Tests (lit-based)

**Type**: `iree_lit_test`
**Framework**: MLIR lit tests with FileCheck
**Location**: Adjacent to source code in `test/` directories
**Naming**: `*_ops.mlir`, `*_folding.mlir`, etc.
**Tool**: Uses `iree-opt` instead of `mlir-opt`

### Runtime Tests (GoogleTest)

**Type**: `iree_cc_test`
**Framework**: GoogleTest
**Location**: Same directory as source, suffixed `_test.cc`
**Include**: Use `iree/testing/gtest.h` instead of gtest headers
**Main**: Link against `iree::testing::gtest_main`

### End-to-End Tests (check framework)

**Type**: `iree_check_test`
**Framework**: Custom check framework (compile + execute as gtest)
**Location**: `tests/e2e/`
**Format**: MLIR modules with exported functions as test cases
**Assertions**: Use `check.expect_*` ops, `util.unfoldable_constant` for test data
**Build**: Requires `iree-test-deps` target
**Example**:
```mlir
func.func @test_floor() {
  %input = util.unfoldable_constant dense<1.1> : tensor<f32>
  %result = "mhlo.floor"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<1.0> : tensor<f32>): tensor<f32>
  return
}
```

### Test Configuration

**CMake**: Functions like `iree_lit_test()`, `iree_cc_test()`, `iree_check_test()` mirror Bazel rules
**Bazel**: Primary BUILD files, converted to CMakeLists.txt automatically
**Naming**: Tests follow `check_backend_driver_src` convention

## Important File Patterns

- `compiler/src/iree/compiler/`: Compiler source
- `runtime/src/iree/`: Runtime source
- `compiler/plugins/`: Input and target backend plugins
- `tests/e2e/`: End-to-end tests
- `build_tools/cmake/`: CMake utilities and helper scripts
- `build_tools/bazel_to_cmake/`: Bazel to CMake conversion
- `.bazelversion`: Required Bazel version (if using Bazel directly)

## Key Concepts

- **HAL (Hardware Abstraction Layer)**: Runtime interface to compute devices
- **VM (Virtual Machine)**: IREE's bytecode VM for compiled programs
- **Flow**: High-level tensor dialect for workload partitioning
- **Stream**: Asynchronous execution and device placement
- **Plugins**: Extensible system for custom input dialects and target backends
- **Target Backend**: Compiler backend (LLVMCPU, CUDA, Vulkan, etc.)
- **HAL Driver**: Runtime driver implementation (local-task, cuda, vulkan, etc.)
- **Check Framework**: E2E testing framework that compiles programs as standalone test binaries
