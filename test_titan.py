import torch
import titan_inference
engine = titan_inference.get_engine("legacy_1.4b_project/titan_checkpoints/phase_sft20_best.pt")
try:
    engine.stream("Hello").__next__()
except Exception as e:
    import traceback
    traceback.print_exc()
