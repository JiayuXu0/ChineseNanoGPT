"""
Sample from a trained model
"""

import os
import torch
from transformers import BertTokenizer
from model import GPTConfig, GPT

device = "cuda"
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

out_dir = "out"
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)

# model
gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
model.load_state_dict(checkpoint["model"])
model = torch.compile(model, mode="reduce-overhead")
model.eval()
model.to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# start = enc.encode("\n")
start = tokenizer("书架上")["input_ids"][1:-1]
print(start)
x = torch.tensor(start, dtype=torch.long, device=device)[None, ...]

for k in range(1):

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = model.generate(x, 300, temperature=0.95, top_k=50)

    print(tokenizer.decode(y[0].tolist()))
    print("---------------")
