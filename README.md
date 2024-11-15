# ChineseNanoGPT

中文的微型GPT2训练

## 数据集

* 中文维基数据集
* 下载地址: [Link](https://pan.baidu.com/s/1v1sw8wb0NUvnSC4NlMuQfg?pwd=ba5n) 提取码: ba5n
* 数据放在 data/wiki_zh 目录下
* 读取数据，并进行token转换预处理并保存

```shell
poetry run python data/prepare.py
```

## 采用poetry进行项目管理

```shell
poetry install
```

## 设置wandb

在[wandb网址](https://wandb.ai/)进行注册，然后再本地端进行登录

```shell
wandb login
# 输入网站上的api-key
```

## TODO

* 增加batch大小
* 增加验证效果
* 全量训练
* 更换token进行验证
* 模型逻辑的梳理
