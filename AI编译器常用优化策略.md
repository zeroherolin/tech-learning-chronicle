# AI编译器常用优化策略

## （前端）计算图优化

### 算子融合（Op Fusion）

将多个算子按模式合并，减少中间存储和内存访问

```python
x = input()
y = relu(matmul(x, W1) + b1)
z = sigmoid(matmul(y, W2) + b2)

# After Fusion

z = fused_mlp_layer(x, [W1, b1, W2, b2])
```

### 布局转换（Layout Transform）

转换张量存储格式，优化访存与计算控制

```python
# NCHW -> NHWC
input = torch.randn(16, 3, 224, 224)  # NCHW
output = input.permute(0, 2, 3, 1)    # NHWC
```

### 内存分配（Memory Allocation）

- 内存类型划分 ：静态内存（编译时分配，如模型参数）；动态内存（运行时按需分配，如中间特征）

- Inplace操作 ：对生命周期已结束的张量就地覆盖，减少内存碎片

- 内存复用与缓冲区共享 ：自动分析张量生命周期，复用不再使用的张量空间，降低峰值内存消耗

- 并行感知分配 ：在不发生并发冲突的前提下，最大化共享内存，提高并行效率

### 常量折叠与常量传播（Constant Folding & Propagation）

编译期计算可定值的表达式，简化运算量；利用常量传播消除不必要分支

```python
a = 2.5 * 4
b = 20
c = a * x + b + 5

# After Folding

a = 10
b = 20

# After Propagation

c = 10 * x + 25

```

### 代数简化（Algebraic Reduction）

用数学恒等简化表达式，提升执行效率

```python
x = a * 0
y = b * 1
z = (c * d) / d

# After Reduction

x = 0
y = b
z = c
```

### 死代码消除（Dead Code Elimination）

去除无侧效、不被后续使用的数据计算，减轻计算与内存负担

```python
a = compute_expensive()
b = used_variable()
c = unused_variable()  # Dead code

# After Elimination

a = compute_expensive()
b = used_variable()
```

### 公共子表达式消除（Common Subexpression Elimination）

提取多次出现的相同计算，复用结果

```python
x = (a + b) * c
y = (a + b) * d

# After Elimination

tmp = a + b
x = tmp * c
y = tmp * d
```

## （后端）算子级优化

### 循环优化

- 循环展开（Loop Unrolling）

展开循环体，减少分支判断，提高流水线利用率

```python
for j in range(2 * n):
    for i in range(m):
        A[j] += B[i]

# After Unrolling

for j in range(0, 2 * n, 2):
    for i in range(m):
        A[j] += B[i]
        A[j+1] += B[i]
```

- 循环分块（Loop Tiling）

将大循环切分为小块，提升L1/L2 Cache命中率，适用于矩阵乘、卷积等

```python
for j in range(n):
    for i in range(m):
        A[j] += B[i]

# After Tiling

for j_o in range(0, n, T):
    for i in range(m):
        for j_i in range(j_o, j_o + T):
            A[j_i] += B[i]
```

- 循环重排（Loop Reorder）

依赖关系允许时调整循环顺序，提升数据局部性或并行性

```python
for i in range(n):
    for j in range(m):
        A[i][j] = B[i][j] * C[i][j]

# After Reorder

for j in range(m):
    for i in range(n):
        A[i][j] = B[i][j] * C[i][j]
```

- 循环融合（Loop Fusion）

将遍历相同数据的多个循环合并，减少多次遍历带来的cache miss

```python
for i in range(0, n + 1):
    A[i] = a[i] + b[i]
    c[i] = 2 * a[i]

for i in range(1, n):
    D[i] = c[i] + a[i]

# After Fusion

A[0] = a[0] + b[0]
c[0] = 2 * a[0]
A[n-1] = a[n-1] + b[n-1]
c[n-1] = 2 * a[n-1]

for i in range(1, n):
    A[i] = a[i] + b[i]
    c[i] = 2 * a[i]
    D[i] = c[i] + a[i]
```

- 循环拆分（Loop Split）

将复杂（带控制流）循环拆为多个子循环，便于并行与优化

```python
for i in range(0, n):
    A[i] = a[i] + b[i]
    c[i] = 2 * a[i]
    if temp[i] > threshold:
        d[i] = a[i]

# After Split

for i in range(0, n):
    A[i] = a[i] + b[i]
    c[i] = 2 * a[i]

for i in range(0, n):
    if temp[i] > threshold:
        d[i] = a[i]
```

### 指令优化

- 向量化（Vectorization）

利用SIMD/SIMT指令（如AVX、ARM Neon、CUDA warp），批量处理数据

- 张量化（Tensorization）

面向高性能TensorCore、矩阵乘等特殊硬件，生成专用指令块

- 指令调度

重排指令避免流水线停顿,提升指令并发度和吞吐量

### 存储优化

- 延迟隐藏（Latency Hiding）

内存操作与计算重叠，最大限度地提高内存和计算资源利用率

- 内存分配（Memory Allocation）

局部变量：开发者定义普通变量时编译器在内存中的栈空间为其分配一段内存 \
全局变量：编译器在内存中的静态存储区分配空间 \
堆变量：开发者用malloc或new在堆上申请一段内存空间

### 并行化与流水线

- 数据并行

按数据块、batch切分任务，多核/多线程/多加速器同时处理不同数据

- 任务/模型并行

针对模型大、无法装入单个设备时的算子/层级分布式拆解

- 流水线调度

流水作业方式分步执行，最大化资源利用率和吞吐量

## Auto-Tuning自动调优

通过自动搜索和试验不同实现方案，找到在目标硬件平台上性能最优的参数配置或代码变体

### 原理

- 预先定义多个实现模板（如不同的循环分块策略、并行划分方案、向量宽度等）
- 编译器基于算子的特征和目标硬件，自动生成多种候选实现
- 在实际设备或模拟器上对这些实现进行运行或性能建模，采集关键性能指标（如延迟、带宽、功耗等）
- 采用启发式搜索、遗传算法、贝叶斯优化等自动算法，遍历或推断最优解
- 反馈循环：使用机器学习模型（如XGBoost）预测性能，加速搜索过程
- 目标：最小化执行时间、内存使用或能量消耗

### 应用方式

- 算子级调优（Operator-level Tuning）：如为一个Conv2D算子自动选取最佳分块大小、线程布局、数据排布方式等
- 图级调优（Graph-level Tuning）：优化整个模型的算子顺序、并行调度、内存布局策略等
- 硬件特定调优：针对不同的CPU、GPU、TPU或自研AI加速器，生成/选择最适配的执行代码
- 离线 vs 在线调优：离线预生成最佳配置，在线根据运行时数据动态调整
- 集成框架：嵌入编译器如TVM或XLA中，作为Pass或插件

### 典型案例

- TVM AutoScheduler：自动机器学习搜索算子调度和kernel参数组合，显著提升算子性能
- TensorRT、XLA、OneDNN等均内置部分自动选择最优内核或参数
- OpenTuner、HyperOpt、Autotune（TensorFlow）：通用或深度学习领域的超参数自动调优器
- Ansor（TVM扩展）：使用规则和ML结合的搜索，针对复杂算子
- AutoTVM：TVM的经典调优器，使用模板和XGBoost模型

### 搜索空间

- 定义：所有可能参数配置的集合，如分块大小、循环顺序、线程数、内存布局等
- 维度：离散（e.g., 线程块大小：16,32,64）或连续（e.g., 学习率），通常高维（数十到数百参数）
- 约束：硬件限制（如寄存器上限、共享内存大小）、依赖关系（参数间互斥）
- 设计策略：手动定义模板或自动生成（如基于算子形状）
- 挑战：爆炸式增长，需要剪枝或采样技术减少评估成本

### 性能评估

- 指标：执行时间（latency）、吞吐量（throughput）、内存使用、功耗、FLOPs利用率
- 方法：实机基准测试（运行代码测量）、模拟器估算（快速但不精确）、代理模型预测（ML-based）
- 反馈机制：将评估结果反馈给搜索算法，迭代优化
- 多目标优化：权衡时间 vs 内存等，使用Pareto前沿选择
- 工具：集成profiler如Nsight或perf，自动化基准脚本

***
🔙 [Go Back](README.md)
