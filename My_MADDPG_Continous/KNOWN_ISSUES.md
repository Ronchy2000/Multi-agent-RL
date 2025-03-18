# Known Issues & Solutions | 已知问题与解决方案

[🇺🇸 English](#english) | [🇨🇳 中文](#chinese)

<a id="english"></a>
## English

This document lists known issues in the project and their solutions.

### Table of Contents
- [Rendering Issues](#rendering-issues)
- [PettingZoo Version Compatibility](#pettingzoo-version-compatibility)
- [Other Common Issues](#other-common-issues)

### Rendering Issues

#### Windows Rendering Unresponsiveness

**Issue**: When using PettingZoo's MPE environment on Windows systems, the rendering window may become unresponsive.

**Solution**:
1. Replace the official `simple_env.py` file with our fixed version:
```bash
# Copy the fixed renderer to your PettingZoo installation path
cp envs/simple_env_fixed_render.py <YOUR_PETTINGZOO_PATH>/pettingzoo/mpe/_mpe_utils/simple_env.py
```

But, I suggest you find the `simple_env.py` file in your PettingZoo installation path and replace it with the fixed version `simple_env_fixed_render.py`. **Copy and paste the code into the file manually.**

2. The key fix is adding proper event handling to ensure it works on all platforms:
```python
# Add event handling to fix rendering issues on Windows
for event in pygame.event.get():
    if event.type == pygame.QUIT:
        pygame.quit()
        return
    if event.type == pygame.WINDOWCLOSE:
        pygame.quit()
        return
pygame.event.pump()  # Ensure the event system runs properly
```
### PettingZoo Version Compatibility
**Issue**: This project requires PettingZoo 1.24.4, but the official PyPI repository only offers version 1.24.3.

**Solution**:
Install version 1.24.4 from GitHub source:
```bash
pip install "pettingzoo[mpe] @ git+https://github.com/Farama-Foundation/PettingZoo.git"
```

Or use the provided installation script:
```python
python utils/setupPettingzoo.py
```

### Other Common Issues
#### Visdom Server Connection Issues

**Issue**: Unable to connect to the Visdom server.

**Solution**:

1. Ensure the Visdom server is running: python -m visdom.server
2. Check if the port is in use, try specifying another port: python -m visdom.server -port 8098
3. Make sure the firewall is not blocking the Visdom service


# Known Issues & Solutions | 已知问题与解决方案
[🇺🇸 English](#english) | 🇨🇳 [中文](#chinese)

<a id="chinese"></a>
## 中文

本文档列出了项目中已知的问题及其解决方案。

### 目录
- [渲染问题](#渲染问题)
- [PettingZoo版本兼容性](#pettingzoo版本兼容性)
- [其他常见问题](#其他常见问题)

### 渲染问题

#### Windows系统渲染无响应

**问题描述**：在Windows系统上，使用PettingZoo的MPE环境时，渲染窗口可能会变得无响应。

**解决方案**：
1. 使用我们修复后的`simple_env.py`文件替换官方版本：
```bash
# 将修复后的渲染器复制到您的PettingZoo安装路径中
cp envs/simple_env_fixed_render.py <YOUR_PETTINGZOO_PATH>/pettingzoo/mpe/_mpe_utils/simple_env.py
```

2. 修复的关键在于添加了适当的事件处理，确保在所有平台上都能正常工作：
```python
# 添加事件处理, 解决windows渲染报错
for event in pygame.event.get():
    if event.type == pygame.QUIT:
        pygame.quit()
        return
    if event.type == pygame.WINDOWCLOSE:
        pygame.quit()
        return
pygame.event.pump()  # 确保事件系统正常运行
```
### PettingZoo版本兼容性
#### 问题描述 

本项目需要PettingZoo 1.24.4版本，但官方PyPI仓库最新版本仅为1.24.3
#### 解决方案
从GitHub源码安装1.24.4版本：
```bash
pip install "pettingzoo[mpe] @ git+https://github.com/Farama-Foundation/PettingZoo.git"
```
或使用提供的安装脚本：
```bash
python utils/setupPettingzoo.py
```

### 其他常见问题

#### Visdom服务器连接问题

**问题描述**：无法连接到Visdom服务器。

**解决方案**：
1. 确保Visdom服务器已启动：`python -m visdom.server`
2. 检查端口是否被占用，可以尝试指定其他端口：`python -m visdom.server -port 8098`
3. 确保防火墙未阻止Visdom服务
