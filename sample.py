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
dtype = torch.float16
out_dir = "out"
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)

# model
gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
model = torch.compile(model, mode="reduce-overhead")
model.load_state_dict(checkpoint["model"])
model.eval()
model.to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# start = enc.encode("\n")
start = tokenizer("北京市包括")["input_ids"][1:-1]
print(start)
x = torch.tensor(start, dtype=torch.long, device=device)[None, ...]

for k in range(1):

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            y = model.generate(x, 300, temperature=0.95, top_k=50)

    print(tokenizer.decode(y[0].tolist()))
    print("---------------")
