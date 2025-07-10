# TVM 深度学习编译器

Code: [https://github.com/apache/tvm](https://github.com/apache/tvm) \
Doc Webpage: [https://tvm.apache.org/docs/](https://tvm.apache.org/docs/)

## 构建TVM环境

### 源码构建

```bash
conda create -n tvm-venv -c conda-forge \
    "llvmdev>=15" \
    "cmake>=3.24" \
    git \
    python=3.11

conda activate tvm-venv
conda install cython

git clone --recursive https://github.com/apache/tvm

mkdir build && cd build
cp ../cmake/config.cmake .

echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake && \
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake && \
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake && \
echo "set(USE_CUDA   ON)" >> config.cmake && \
echo "set(USE_METAL  OFF)" >> config.cmake && \
echo "set(USE_VULKAN OFF)" >> config.cmake && \
echo "set(USE_OPENCL OFF)" >> config.cmake && \
echo "set(USE_CUBLAS ON)" >> config.cmake && \
echo "set(USE_CUDNN  ON)" >> config.cmake && \
echo "set(USE_CUTLASS OFF)" >> config.cmake

cmake ..
cmake --build . --parallel $(nproc)

export TVM_LIBRARY_PATH=$(pwd)
pip install -e ../python
```

### 极简安装

```bash
pip install apache-tvm
```

### 验证安装

```bash
# Check version
python -c "import tvm; print(tvm.__version__)"

# Locate TVM Python package
python -c "import tvm; print(tvm.__file__)"

# Confirm which TVM library is used
python -c "import tvm; print(tvm.ffi)"
# python -c "import tvm; print(tvm._ffi)"

# Check device detection (does not work with pip-installed prebuilt version)
python -c "import tvm; print(tvm.metal().exist)"
python -c "import tvm; print(tvm.cuda().exist)"
python -c "import tvm; print(tvm.vulkan().exist)"

# Reflect TVM build option
python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
```

***
🔙 [Go Back](README.md)
