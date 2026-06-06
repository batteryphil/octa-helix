import torch
import deepspeed
from torch import nn
import json

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(10, 10)
    def forward(self, x): return self.l(x)

with open('ds_dummy.json', 'w') as f:
    json.dump({
        "train_batch_size": 1,
        "zero_optimization": {"stage": 3},
        "bfloat16": {"enabled": True}
    }, f)

model = Dummy()
engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config='ds_dummy.json'
)

print("Engine type:", type(engine))
try:
    engine.save_16bit_model("dummy.pt")
    print("Success!")
except Exception as e:
    print("Error:", type(e), e)
