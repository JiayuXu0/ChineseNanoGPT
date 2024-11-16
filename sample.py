"""
用于从训练好的模型中采样生成文本
"""

import os
import torch
from transformers import BertTokenizer
from model import GPTConfig, GPT


class Config:
    """配置类"""

    device = "cuda"
    seed = 1337
    out_dir = "out"
    max_tokens = 300
    temperature = 0.95
    top_k = 50


def setup_environment() -> None:
    """设置环境配置"""
    torch.manual_seed(Config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_model() -> tuple[GPT, BertTokenizer]:
    """加载模型和分词器"""
    try:
        ckpt_path = os.path.join(Config.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=Config.device)

        # 初始化模型
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        model.load_state_dict(checkpoint["model"])
        model = torch.compile(model, mode="reduce-overhead")
        model.eval()
        model.to(Config.device)

        # 加载分词器
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")


def generate_text(model: GPT, tokenizer: BertTokenizer, prompt: str) -> str:
    """生成文本"""
    # 对输入文本进行编码
    start = tokenizer(prompt)["input_ids"][1:-1]
    x = torch.tensor(start, dtype=torch.long, device=Config.device)[None, ...]

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = model.generate(
                x,
                Config.max_tokens,
                temperature=Config.temperature,
                top_k=Config.top_k,
            )

    return tokenizer.decode(y[0].tolist())


def main():
    """主函数"""
    setup_environment()
    model, tokenizer = load_model()

    prompt = "书架上"
    generated_text = generate_text(model, tokenizer, prompt)
    print(f"输入提示: {prompt}")
    print("生成结果:")
    print(generated_text)
    print("-" * 50)


if __name__ == "__main__":
    main()
