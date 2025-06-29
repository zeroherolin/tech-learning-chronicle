# 常见AI模型分类

## 1 卷积神经网络（CNN）

| 模型名称         | 提出年份 | 核心贡献机构       | 主要特点                   | 典型应用场景       |
|--------------|------|--------------|------------------------|--------------|
| LeNet        | 1998 | Yann LeCun团队 | 首个成功应用的CNN架构           | 手写数字识别       |
| AlexNet      | 2012 | 多伦多大学        | 深度CNN突破，ReLU和Dropout应用 | ImageNet图像分类 |
| VGGNet       | 2014 | 牛津大学         | 统一3×3卷积核堆叠架构           | 图像特征提取       |
| Inception    | 2014 | Google       | 多分支并行结构（1×1,3×3,5×5卷积） | 大规模图像分类      |
| ResNet       | 2015 | 微软研究院        | 残差连接解决梯度消失             | 深度网络训练       |
| MobileNet    | 2017 | Google       | 深度可分离卷积轻量化             | 移动端图像识别      |
| EfficientNet | 2019 | Google       | 复合缩放优化模型尺寸             | 资源受限环境部署     |

## 2 循环神经网络（RNN）

| 模型名称        | 提出年份 | 核心贡献机构      | 主要特点            | 典型应用场景  |
|-------------|------|-------------|-----------------|---------|
| Vanilla RNN | 1986 | 多机构         | 基础循环结构          | 简单序列建模  |
| BiRNN       | 1997 | Schuster等   | 双向信息处理          | 上下文依赖建模 |
| LSTM        | 1997 | Hochreiter等 | 门控机制解决长程依赖      | 长序列预测   |
| GRU         | 2014 | Cho等        | 简化版LSTM（更新/重置门） | 高效序列处理  |
| SRU         | 2017 | 清华大学等       | 高速通道+并行计算       | 实时语音识别  |

## 3 强化学习模型

| 模型名称   | 提出年份 | 核心贡献机构   | 主要特点       | 典型应用场景  |
|--------|------|----------|------------|---------|
| DQN    | 2013 | DeepMind | 深度Q学习+经验回放 | Atari游戏 |
| A3C    | 2016 | DeepMind | 异步并行策略优化   | 实时控制    |
| PPO    | 2017 | OpenAI   | 近端策略优化稳定训练 | 机器人控制   |
| SAC    | 2018 | Berkeley | 最大熵强化学习    | 复杂环境决策  |
| MuZero | 2019 | DeepMind | 无模型规划      | 围棋/象棋   |

## 4 生成对抗网络（GAN）

| 模型名称     | 提出年份 | 核心贡献机构    | 主要特点            | 典型应用场景   |
|----------|------|-----------|-----------------|----------|
| DCGAN    | 2015 | Radford等  | 深度卷积GAN稳定训练     | 图像生成     |
| WGAN     | 2017 | Arjovsky等 | Wasserstein距离优化 | 训练稳定性提升  |
| CycleGAN | 2017 | Berkeley  | 无配对图像转换         | 风格迁移     |
| StyleGAN | 2019 | NVIDIA    | 风格混合+微调控制       | 高分辨率人脸生成 |
| BigGAN   | 2018 | DeepMind  | 大规模图像生成         | 复杂场景生成   |

## 5 图神经网络（GNN）

| 模型名称      | 提出年份 | 核心贡献机构      | 主要特点      | 典型应用场景 |
|-----------|------|-------------|-----------|--------|
| GCN       | 2016 | Kipf等       | 图卷积操作基础架构 | 节点分类   |
| GraphSAGE | 2017 | Hamilton等   | 归纳式邻居采样聚合 | 大规模图处理 |
| GAT       | 2018 | Velickovic等 | 注意力机制加权聚合 | 异构图处理  |
| GIN       | 2019 | Xu等         | 图同构判别最强架构 | 分子属性预测 |
| STGCN     | 2018 | 北大等         | 时空图卷积     | 交通流量预测 |

## 6 Transformer架构

| 模型名称             | 提出年份      | 核心贡献机构  | 主要特点         | 典型应用场景 |
|------------------|-----------|---------|--------------|--------|
| Transformer      | 2017      | Google  | 自注意力机制基础架构   | 机器翻译   |
| BERT             | 2018      | Google  | 双向预训练+掩码语言模型 | 自然语言理解 |
| GPT-3/4          | 2020/2023 | OpenAI  | 超大规模生成式预训练   | 文本生成   |
| ViT              | 2020      | Google  | 图像分块处理替代CNN  | 图像分类   |
| T5               | 2020      | Google  | 文本到文本统一框架    | 多任务NLP |
| Swin Transformer | 2021      | 微软亚洲研究院 | 层级滑动窗口注意力    | 密集预测任务 |

## 7 扩散模型（Diffusion）

| 模型名称             | 提出年份      | 核心贡献机构            | 主要特点        | 典型应用场景  |
|------------------|-----------|-------------------|-------------|---------|
| DDPM             | 2020      | Ho等               | 去噪扩散概率基础模型  | 图像生成    |
| Stable Diffusion | 2022      | CompVis/Stability | 潜在空间扩散+文本引导 | 文生图     |
| DALL-E 2/3       | 2022/2023 | OpenAI            | CLIP引导扩散模型  | 多模态生成   |
| Imagen           | 2022      | Google            | 大语言模型引导     | 高保真图像生成 |
| Make-A-Video     | 2022      | Meta              | 时空扩散模型      | 文生视频    |

## 8 其他模型

| 模型类型       | 模型名称         | 提出年份 | 核心特点        | 典型应用场景  |
|------------|--------------|------|-------------|---------|
| 深度信念网络 | DBN          | 2006 | 多层RBM堆叠预训练  | 特征提取    |
| 自编码器   | VAE          | 2013 | 概率生成式自编码    | 数据生成/降维 |
| 胶囊网络   | CapsNet      | 2017 | 向量神经元保留空间关系 | 姿态不变性识别 |
| 神经微分方程 | Neural ODE   | 2018 | 连续深度建模      | 不规则时序数据 |
| 混合专家模型 | Mixtral 8x7B | 2023 | 稀疏激活降低计算成本  | 大语言模型   |

***
🔙 [Go Back](README.md)
