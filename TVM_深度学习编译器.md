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

git clone --recursive https://github.com/apache/tvm


```

### 极简安装

```bash

```

### 验证安装

```bash
# Check version
python -c "import tvm; print(tvm.__version__)"

# Locate TVM Python package
python -c "import tvm; print(tvm.__file__)"

# Confirm which TVM library is used
python -c "import tvm; print(tvm._ffi)"

# Check device detection (does not work with pip-installed prebuilt version)
python -c "import tvm; print(tvm.metal().exist)"
python -c "import tvm; print(tvm.cuda().exist)"
python -c "import tvm; print(tvm.vulkan().exist)"

# Reflect TVM build option
python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
```

***
🔙 [Go Back](README.md)
