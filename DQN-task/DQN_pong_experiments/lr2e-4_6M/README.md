# 项目说明

## 环境配置

1. **Python 版本**  
   本项目使用 **Python 3.10**。请确保你创建的虚拟环境使用该版本。

2. **安装 PyTorch**  
   根据你的硬件设备选择安装命令：
   
   - **有 GPU（CUDA）**：
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
     > 将 `cu118` 替换为你显卡对应的 CUDA 版本`。

   - **CPU-only**：
     ```bash
     pip install torch torchvision torchaudio
     ```

3. **安装依赖**  
   使用项目提供的 `requirements.txt` 安装其他依赖：
   ```bash
   pip install -r requirements.txt
   ``` 
   主要下载的库为gymnasium，stable_baselines3，numpy，ale-py，tensorboard等，按照python 3.10下载默认版本即可
如果环境配置有问题可以随时在群里询问
