import math
import os
import time
from datetime import datetime
import gc
import numpy as np
import torch
import wandb

from model import GPT, GPTConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:128,expandable_segments:True"
)

# -----------------------------------------------------------------------------
# 设置参数
# 输入/输出
out_dir = "out"
eval_interval = 2000
log_interval = 1
# wandb日志设置
wandb_log = True
wandb_project = "Chinese-GPT"
wandb_run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# 数据相关
batch_size = 8  # 从10降到4
block_size = 1024  # 从1024降到512
# 模型相关
device = "cuda"
init_from = (
    "scratch"  # 可选值：'scratch'(从头训练) 或 'resume'(继续训练) 或 'gpt2*'
)
dropout = 0
n_layer = 12  # 从12降到8
n_head = 12  # 从12降到8
n_embd = 768  # 从768降到512
# adamw优化器参数
gradient_accumulation_steps = 5 * 8
learning_rate = 6e-4  # 最大学习率
max_iters = 600000  # 训练总迭代次数
weight_decay = 1e-1
betas = (0.9, 0.95)
# 学习率衰减设置
decay_lr = True  # 是否使用学习率衰减
warmup_iters = 2000  # 预热步数
lr_decay_iters = 600000  # 学习率衰减的总步数
min_lr = 6e-5  # 最小学习率
grad_clip = 1.0  # 梯度裁剪阈值
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 简易数据加载器
data_dir = "data"
train_data = np.memmap(
    os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
)
val_data = np.memmap(
    os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy(
                (data[i : i + block_size]).astype(np.int64)  # noqa
            )  # noqa
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                (data[i + 1 : i + 1 + block_size]).astype(np.int64)  # noqa
            )
            for i in ix
        ]
    )
    x, y = x.to(device), y.to(device)
    return x, y


# 模型初始化
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    dropout=dropout,
)
if init_from == "scratch":
    # 从头开始训练
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
model.to(device)

# 使用 torch.compile 编译模型
model = torch.compile(model, mode="reduce-overhead")


@torch.no_grad()
def estimate_loss(eval_iters=50):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    # 评估后清理显存
    gc.collect()  # 先进行Python的垃圾回收
    torch.cuda.empty_cache()  # 释放PyTorch的CUDA缓存
    return out


# 优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, betas)


# 学习率衰减调度器（带预热的余弦衰减）
def get_lr(iter):
    # 1) 在预热步数内进行线性预热
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # 2) 如果迭代次数超过衰减步数，返回最小学习率
    if iter > lr_decay_iters:
        return min_lr
    # 3) 在预热和最大衰减步数之间，使用余弦衰减到最小学习率
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 范围在0到1之间
    return min_lr + coeff * (learning_rate - min_lr)


# 日志记录
if wandb_log:
    wandb.init(project=wandb_project, name=wandb_run_name)
    wandb.config = {
        "batch_size": batch_size,
        "block_size": block_size,
        "learning_rate": learning_rate,  # 待办：记录所有其他参数
    }

# 训练循环
iter_num = 0
num_tokens = 0
best_val_loss = 1e9
t0 = time.time()
# 在训练循环之前初始化 scaler
scaler = torch.amp.GradScaler("cuda")
while True:
    # 根据迭代次数决定学习率
    if decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    else:
        lr = learning_rate

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss \
            {losses['val']:.4f}"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "num_tokens": num_tokens,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                }
            )
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                }
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                gc.collect()
                torch.cuda.empty_cache()

    # 修改训练步骤以支持梯度累积
    # 只在累积完成后更新优化器
    is_accumulation_step = iter_num % gradient_accumulation_steps != 0

    X, Y = get_batch("train")
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(X, Y)
        # 根据梯度累积步数缩放损失
        loss = loss / gradient_accumulation_steps

    # 使用scaler进行反向传播
    scaler.scale(loss).backward()

    # 只在非累积步骤时更新参数
    if not is_accumulation_step:
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # 使用scaler更新优化器
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = (
            loss.item() * gradient_accumulation_steps
        )  # 还原实际损失值用于显示
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1
    num_tokens += X.numel()

    # 终止条件
    if iter_num >= max_iters:
        break
