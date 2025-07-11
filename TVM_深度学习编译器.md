# TVM 深度学习编译器

<img src="assets/tvm-logo-small.png" width=150/>

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

# Check device detection
python -c "import tvm; print(tvm.metal().exist)"
python -c "import tvm; print(tvm.cuda().exist)"
python -c "import tvm; print(tvm.vulkan().exist)"

# Reflect TVM build option
python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
```

## TVM总体流

<img src="assets/tvm_overall_flow.svg" width=600/>

- 构建或导入模型：构建神经网络模型或从其他框架（如PyTorch、ONNX）导入预训练模型，并创建TVM IRModule，其中包含编译所需的所有信息，包括用于计算图的高级Relax函数和用于张量程序的低级TensorIR函数

- 执行可组合优化：执行一系列优化转换，如图形优化、张量程序优化和库调度

- 构建和通用部署：将优化后的模型构建为可部署到通用运行时的模块，并在不同的设备（如CPU、GPU或其他加速器）上执行

## CPU基础测试

```python
import tvm
from tvm import relax
from tvm.relax.frontend import nn
import numpy as np


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


mod, param_spec = MLPModel().export_tvm(
    spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)
mod.show()

mod = relax.get_pipeline("zero")(mod)

target = tvm.target.Target("llvm")
device = tvm.cpu()

ex = relax.build(mod, target)
vm = relax.VirtualMachine(ex, device)

params = []
for _, param_info in param_spec:
    param_np = np.random.rand(*param_info.shape).astype("float32")
    params.append(tvm.nd.array(param_np, device=device))

data = np.random.rand(1, 784).astype("float32")
tvm_data = tvm.nd.array(data, device=device)

out = vm["forward"](tvm_data, *params)
print(out.numpy())
```

## CUDA基础测试

```python
import tvm
from tvm import te
import numpy as np

n = 16
A = te.placeholder((n, n), name="A")
B = te.placeholder((n, n), name="B")
k = te.reduce_axis((0, n), name="k")
C = te.compute((n, n), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

func = te.create_prim_func([A, B, C])
mod = tvm.IRModule({"main": func})
mod.show()

sch = tvm.tir.Schedule(mod)
block = sch.get_block("C", func_name="main")
loops = sch.get_loops(block)

i, j = loops[:2]
io, ii = sch.split(i, factors=[None, 4])
jo, ji = sch.split(j, factors=[None, 4])
sch.bind(io, "blockIdx.x")
sch.bind(jo, "blockIdx.y")
sch.bind(ii, "threadIdx.x")
sch.bind(ji, "threadIdx.y")

target = tvm.target.Target("cuda")
device = tvm.cuda(0)

cuda_mod = tvm.build(sch.mod, target=target)

a_np = np.ones((n, n)).astype(np.float32)
b_np = np.ones((n, n)).astype(np.float32)
a = tvm.nd.array(a_np, device=device)
b = tvm.nd.array(b_np, device=device)
c = tvm.nd.array(b_np, device=device)

cuda_mod(a, b, c)
print(c.numpy())
```

***
🔙 [Go Back](README.md)
