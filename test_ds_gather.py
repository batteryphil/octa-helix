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

import os
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29501'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

model = Dummy()
engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config='ds_dummy.json'
)

with deepspeed.zero.GatheredParameters(engine.module.parameters()):
    sd = engine.module.state_dict()
    print("Gathered successfully! Keys:", sd.keys())

