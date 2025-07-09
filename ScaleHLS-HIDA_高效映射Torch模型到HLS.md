# ScaleHLS-HIDA 高效映射Torch模型到HLS

Code: [https://github.com/UIUC-ChenLab/ScaleHLS-HIDA](https://github.com/UIUC-ChenLab/ScaleHLS-HIDA)

## 构建ScaleHLS环境

```bash
git clone https://github.com/UIUC-ChenLab/ScaleHLS-HIDA.git --recursive

cd ScaleHLS-HIDA/polygeist/llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

ninja

cd ../../..
# edit externals/ScaleHLS-HIDA/CMakeLists.txt and add the following: 
set(LLVM_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/polygeist/llvm-project/build")
set(LLVM_DIR "${LLVM_BUILD_DIR}/lib/cmake/llvm")
set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir")

mkdir build && cd build
cmake -G Ninja .. \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

ninja scalehls-opt scalehls-translate
```
***
🔙 [Go Back](README.md)
