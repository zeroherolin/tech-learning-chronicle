# Tech Learning Chronicle

## AI编译器相关

[AI编译器基础](AI编译器基础.md)

    Compiler | LLVM | GPGPU | TVM | XLA | MLIR | IREE | Triton

[AST抽象语法树](AST抽象语法树.md) [Updating...]

    语法树结构 | 解析过程 | 在编译器中的应用 | 操作与遍历 | 类型检查 | 优化转换

[编译原理](编译原理.md) [Updating...]

    前端 | 中间表示 | 后端 | 词法分析 | 语法分析 | 语义分析 | 代码生成 | 错误处理

[AI编译器常用优化策略](AI编译器常用优化策略.md)

    前端优化： 算子融合 | 布局转换 | 内存分配 | 常量折叠与常量传播
             代数简化 | 死代码消除（DCE） | 公共子表达式消除（CSE）
    后端优化： 循环优化 | 指令优化 | 存储优化 | 并行化与流水线
    Auto-Tuning自动调优： 原理 | 应用方式 | 典型案例 | 搜索空间 | 性能评估

[JIT和AOT](JIT和AOT.md) [Updating...]

    JIT原理 | AOT原理 | 混合模式 | 示例框架

[TVM 深度学习编译器](TVM深度学习编译器.md)

    构建TVM环境 | TVM总体流 | Relax IR | 算子定义 | 部署示例 | AutoTVM

[LLVM & MLIR 基础](LLVM_and_MLIR基础.md) [Updating...]

    构建MLIR环境 | MLIR Opt 测试 | MLIR EmitC 测试 | Pass机制

[MLIR Dialect 体系](MLIR_Dialect体系.md)

    核心/基础 | 控制流与数据流 | 数据结构/容器 | 数值计算 | 中高层抽象/算子
    高层AI/ML相关 | 硬件相关和低层 | 编译工具链 | 并行与分布式

[Torch-MLIR 连接PyTorch和MLIR](Torch-MLIR连接PyTorch和MLIR.md) [Updating...]

    构建Torch-MLIR环境 | Torch-MLIR测试 | TorchScript到MLIR | 优化Pass | 后端集成 | 性能基准

[ScaleHLS-HIDA 高效映射Torch模型到HLS](ScaleHLS-HIDA高效映射Torch模型到HLS.md) [Updating...]

    构建ScaleHLS环境 | Torch到HLS映射 | 量化与优化 | FPGA部署 | 性能分析

[IREE 端到端编译器运行时系统](IREE端到端编译器运行时系统.md) [Updating...]

[Triton DSL 定义高性能算子](Triton_DSL定义高性能算子.md) [Updating...]

[AI编译器Runtime](AI编译器Runtime.md) [Updating...]

[主流AI编译器框架对比](主流AI编译器框架对比.md) [Updating...]

## AI模型相关

[常见AI模型分类](常见AI模型分类.md)

    卷积神经网络（CNN） | 循环神经网络（RNN） | 强化学习模型 | 生成对抗网络（GAN）
    图神经网络（GNN） | Transformer架构 | 扩散模型（Diffusion） | 其他模型

[NN模型的Pytorch实现](https://github.com/zeroherolin/pytorch-nn) [Updating...]

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

[ONNX生态](ONNX生态.md) [Updating...]

    协议缓冲区 | 算子集 | Runtime API | 优化工具 | MLIR转换

[TensorRT-LLM推理库](TensorRT-LLM推理库.md) [Updating...]

[vLLM推理框架](vLLM推理框架.md) [Updating...]

[模型优化技术](模型优化技术.md) [Updating...]

    量化(PTQ/QAT) | 剪枝(结构化/非结构化) | 知识蒸馏 | 低秩分解 | 模型编译

[案例研究：ViT端到端部署与优化](案例研究_ViT端到端部署与优化.md) [Updating...]

## 处理器架构相关

[NVIDIA GPU架构与CUDA核心](NVIDIA_GPU架构与CUDA核心.md) [Updating...]

    宏观架构： 芯片架构 | SM核心结构 | 执行模型 | 资源分配
    内存子系统： 物理内存 | 缓存体系 | 共享内存机制 | 统一内存系统
    PTX虚拟GPU架构： PTX指令集 | 编译流程 | 关键指令 | 优化接口

[现代CPU架构与SIMD](现代CPU架构与SIMD.md) [Updating...]

## 并行与分布式系统

[深度学习分布式策略](深度学习分布式策略.md) [Updating...]

    数据并行 | 模型并行 | 混合并行

[NCCL通讯库](NCCL通讯库.md) [Updating...]

[MPI并行计算](MPI并行计算.md) [Updating...]

## 操作系统

[Linux内核基础](Linux内核基础.md) [Updating...]

## 编程语言和工具

[C内存管理](C内存管理.md) [Updating...]

    栈内存 | 堆内存

[C编译工具](C编译工具.md)

    gcc | make & Makefile | cmake & CMakeLists | ninja

[性能分析工具](性能分析工具.md) [Updating...]

    CPU侧： perf | prof
    GPU侧： Nvidia Nsight Systems | Nsight Compute
    内存分析： Valgrind

[Pybind绑定生成器](Pybind绑定生成器.md)

    准备工作 | 使用示例 | 编译（gcc/cmake） | Python测试

[Nvidia CUDA 基础](Nvidia_CUDA基础.md) [Updating...]

    核函数 | 线程模型 | 设备管理 | 错误处理 | 内存模型和内存管理
    流 | 事件 | 原子操作 | 性能优化策略 | 高级并行模式 | 多GPU编程 | CUDA库 | 调试工具 | CMake编译

## 附录： Linux软件安装

[Ubuntu安装Docker](Ubuntu安装Docker.md)

[Ubuntu安装Nvidia驱动、CUDA、cuDNN、Pytorch](Ubuntu安装Nvidia驱动、CUDA、cuDNN、Pytorch.md)
