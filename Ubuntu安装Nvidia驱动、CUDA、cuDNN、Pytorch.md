# Ubuntu安装Nvidia驱动、CUDA、cuDNN、Pytorch

## 安装驱动

- 通过apt安装

```bash
sudo apt install nvidia-driver-535-server
```

重启PC，选择 enroll mok -> continue -> yes -> 输入密码 -> reboot

- 查看打印信息

```bash
nvidia-smi
```

## 安装CUDA

- 下载并执行安装程序

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo chmod +x cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run
```

- 设置环境变量

```bash
sudo nano ~/.bashrc
## 添加
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source ~/.bashrc
```

- 查看

```bash
nvcc --version
```

## 安装cuDNN

- 执行安装

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cudnn-cuda-12
```

## 安装Pytorch

- 创建虚拟环境

```bash
conda create -n torch python=3.12
conda activate torch
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```

- 测试

```python
import torch
import torch.utils
import torch.utils.cpp_extension


print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.utils.cpp_extension.CUDA_HOME)
```

***
🔙 [Go Back](README.md)
