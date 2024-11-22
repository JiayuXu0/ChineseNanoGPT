"""
Sample from a trained model
"""

import os
import torch
from transformers import BertTokenizer
from model import GPTConfig, GPT
from transformers import AutoTokenizer
device = "cpu"
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
model = torch.compile(model)
model.load_state_dict(checkpoint["model"])
model.eval()
model.to(device)
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
# start = enc.encode("\n")
start = tokenizer("今天天气怎么样？")["input_ids"][1:-2]
print(start)
x = torch.tensor(start, dtype=torch.long, device=device)[None, ...]

for k in range(1):

    with torch.no_grad():
        with torch.amp.autocast(device_type="cpu", dtype=dtype):
            y = model.generate(x, 300, temperature=0.8, top_k=50)

    print(tokenizer.decode(y[0].tolist()))
    print("---------------")
