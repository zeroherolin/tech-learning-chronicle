# Tech Learning Chronicle

## AI编译器相关

[AI编译器基础](AI编译器基础.md)

    Compiler | LLVM | GPGPU | TVM | XLA

[AST抽象语法树](AST抽象语法树.md) [Updating...]

[常用IR优化](常用IR优化.md) [Updating...]

    计算图优化： 算子融合 | 常量折叠 | 死代码消除（DCE） | 公共子表达式消除（CSE）
    算子级优化： 循环优化（展开、分块） | 内存访问优化 | 并行化策略
    硬件特定优化： 指令集映射 | 张量核心 | 线程/块调度 | 带宽优化

[JIT和AOT](JIT和AOT.md) [Updating...]

[LLVM MLIR 基础测试](LLVM_MLIR_基础测试.md)

    构建MLIR环境 | MLIR Opt 测试 | MLIR EmitC 测试

[MLIR Dialect 体系](MLIR_Dialect_体系.md)

    核心/基础 | 控制流与数据流 | 数据结构/容器 | 数值计算 | 中高层抽象/算子
    高层AI/ML相关 | 硬件相关和低层 | 编译工具链 | 并行与分布式

[Torch-MLIR 连接PyTorch和MLIR](Torch-MLIR_连接PyTorch和MLIR.md)

    构建Torch-MLIR环境 | Torch-MLIR测试

[ScaleHLS-HIDA 高效映射Torch模型到HLS](ScaleHLS-HIDA_高效映射Torch模型到HLS.md) [Updating...]

    构建ScaleHLS环境 |

[IREE 端到端编译器运行时系统](IREE_端到端编译器运行时系统.md) [Updating...]

[Halide 解耦算法描述](Halide_解耦算法描述.md) [Updating...]

[TVM 深度学习编译器](TVM_深度学习编译器.md) [Updating...]

    构建TVM环境 |

[Triton DSL 定义高性能算子](Triton_DSL_定义高性能算子.md) [Updating...]

[Allo Accelerator Design Language](Allo_Accelerator_Design_Language.md)

    构建Allo环境（宿主机或Docker） | Allo测试 | 示例：GEMM | 示例：整数输出稳态脉动阵列 | Allo原语

## AI模型相关

[常见AI模型分类](常见AI模型分类.md)

    卷积神经网络（CNN） | 循环神经网络（RNN） | 强化学习模型 | 生成对抗网络（GAN）
    图神经网络（GNN） | Transformer架构 | 扩散模型（Diffusion） | 其他模型

[nn模型的Pytorch实现](https://github.com/zeroherolin/pytorch-nn) [Updating...]

    Basic： Mnist（手写数字识别）
    CNN： VGG19（图像分类） | ResNet（图像分类）
    RNN： LSTM（文本生成） | GRU（情感分类）
    RL： DQN（游戏控制）
    GAN： CGAN（图像生成） | DCGAN（人脸生成）
    Transformer： Transformer（机器翻译） | ViT（图像分类）

[Transformer模型理论](Transformer模型理论.md)

    输入嵌入 | 位置编码 | 自注意力 | 多头注意力
    位置式前馈神经网络 | 残差连接 | 编码器 | 解码器 | 输出层

[Vision Transformer (ViT) 模型理论](ViT模型理论.md) [Updating...]

[TensorRT-LLM推理库](TensorRT-LLM推理库.md) [Updating...]

[vLLM推理框架](vLLM推理框架.md) [Updating...]

[模型优化技术](模型优化技术.md) [Updating...]

    模型量化 | 模型压缩

## 并行与分布式系统

[NCCL通讯库](NCCL通讯库.md) [Updating...]

[MPI并行计算](MPI并行计算.md) [Updating...]

## 编程工具

[C编译工具](C编译工具.md)

    gcc | make & Makefile | cmake & CMakeLists | ninja

[Pybind绑定生成器](Pybind绑定生成器.md)

    准备工作 | 使用示例 | 编译（gcc/cmake） | Python测试

[Nvidia CUDA 基础](Nvidia_CUDA_基础.md) [Updating...]

    核函数 | 线程模型 | 设备管理 | 错误处理 | 内存模型和内存管理
    流 | 事件 | 原子操作 | 性能优化策略 | 高级并行模式 | 多GPU编程 | CUDA库 | 调试工具 | CMake编译

## Linux软件安装

[Ubuntu安装Docker](Ubuntu安装Docker.md)

[Ubuntu安装Nvidia驱动、CUDA、Pytorch](Ubuntu安装Nvidia驱动、CUDA、Pytorch.md)

## 其他

[Raspberry Pi 5 & Arm64](Linux_Arm64.md)

    Arm64安装Docker | Arm64编译XDMA驱动
