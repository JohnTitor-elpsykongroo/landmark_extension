# 本机训练入口预检（2026-09-17）

本机为 Windows、RTX 5050 Laptop GPU（8 GB）、Python 3.10.21。目标环境仍是 Linux、RTX 5090、CUDA 13.0。此记录仅证明本机入口和数据路径的进度，不代表目标机器或完整训练已验证。

在 `3dteethland` 根目录创建了独立 `.venv`，安装了 pip、setuptools、wheel、NumPy、PyYAML 和 Ninja。PyTorch 等训练依赖尚未装入该 `.venv`。本机已有环境提供 PyTorch 2.11.0+cu128 等包，曾临时借用它们进行路径检查；测试后已恢复 `.venv` 的独立隔离。正式环境配置以 [LINUX_TRAINING_CN.md](LINUX_TRAINING_CN.md) 为准。

两例有位点标注的真实数据副本用于测试：`013TXGFK_upper` 训练、`0140W3ND_upper` 验证。配置、数据视图、日志及构建尝试保留在本地 `local_smoke/20260917T133400Z/`，已被 Git 忽略；原始数据未修改。

测试入口（在项目根目录执行）如下，`launcher.py` 只在本地为缺失的 `pointops` 注册会明确报错的占位函数，用来检查到首次调用 CUDA 算子之前的流程：

```powershell
.\.venv\Scripts\python.exe landmark_extension\local_smoke\20260917T133400Z\launcher.py
```

首次运行发现原 `_files` 在 Windows 路径上筛得 0 例，随后发现原 `TeethLandDataModule.setup()` 创建 `TeethLandDataset` 时没有传 `norm`。两处均只在扩展目录修复。修复后日志显示 `Total number of landmark-annotated files: 2`，训练/验证数据集加载、缓存创建、模型初始化均成功，进入首个验证 batch；首个 KPConv 邻域查询因占位的 `pointops` 报错。**路径与数据加载通过；真实 CUDA 前向和反向训练未通过验证。**

在扩展目录的源码副本上尝试编译真实 `pointops`，失败原因是本机 `nvcc` 为 CUDA 13.0，而现有 PyTorch 使用 CUDA 12.8；本机也未找到 MSVC `cl.exe`。官方 cu130 PyTorch wheel 约 1.9 GB，下载速度很低，本次中止下载。因此无法在这台 Windows 电脑上完成真实训练闭环。目标 Linux 机器需要安装与 CUDA 13.0 相符的 PyTorch 和可用 C++/CUDA 编译工具，然后按主文档完成第一轮真实训练及 checkpoint 检查。
