# ChineseNanoGPT

基于nanoGPT架构的中文语言模型训练实现。本项目受 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 启发,实现了对中文的支持。

## 功能特点

- 支持中文维基百科数据集训练
- 使用BERT中文分词器
- 支持混合精度训练
- 集成Wandb可视化训练过程
- 支持模型断点续训
- 自动学习率调度

## 环境配置

1. 安装依赖管理工具poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. 安装项目依赖:

```bash
poetry install
```

3. 配置Wandb:

- 在[wandb.ai](https://wandb.ai/)注册账号
- 本地登录:

```bash
wandb login
```

## 数据准备

1. 下载中文维基数据集:

- 下载地址: [百度网盘](https://pan.baidu.com/s/1v1sw8wb0NUvnSC4NlMuQfg?pwd=ba5n)
- 提取码: ba5n

2. 解压数据到 `data/wiki_zh` 目录

3. 数据预处理:

```bash
poetry run python data/prepare.py
```

## 训练模型

```bash
poetry run python train.py
```

## 生成文本

```bash
poetry run python sample.py
```

## 开发计划

- [ ] 优化训练超参数配置
- [ ] 抽取配置到独立config文件
- [ ] 支持更大batch size训练
- [ ] 增加模型验证指标
- [ ] 支持全量数据训练
- [ ] 验证不同tokenizer效果
- [ ] 梳理模型核心逻辑
- [ ] 支持LLaMA架构
- [ ] 支持C++推理

## 技术细节

- 模型架构: GPT-2
- 分词器: BERT中文分词器
- 训练框架: PyTorch
- 显存优化: 混合精度训练
- 训练监控: Wandb
- 数据集: 中文维基百科

## 贡献

欢迎提交Issue和Pull Request!

## 许可证

MIT License
