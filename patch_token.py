import os

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # We want to replace `streaming=True)` with `streaming=True, token=HF_TOKEN)`
    # but only if it doesn't already have the token.
    # We know the token variable is HF_TOKEN or _HF_TOKEN in these scripts.
    import re
    # Match `streaming=True)` and replace
    new_content = re.sub(r'streaming=True\)', r'streaming=True, token=HF_TOKEN)', content)
    # Match `streaming=True )` just in case
    new_content = re.sub(r'streaming=True \)', r'streaming=True, token=HF_TOKEN)', new_content)
    
    # Let's fix _HF_TOKEN to HF_TOKEN if needed
    if 'HF_TOKEN = os.environ.get' not in new_content and '_HF_TOKEN = os.environ.get' in new_content:
        new_content = new_content.replace('_HF_TOKEN = os.environ.get', 'HF_TOKEN = os.environ.get')
        new_content = new_content.replace('_HF_TOKEN', 'HF_TOKEN')

    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"Patched {filepath}")

patch_file('/home/phil/.gemini/antigravity/scratch/analysis_project/phase_1_deepspeed_trainer.py')
patch_file('/home/phil/.gemini/antigravity/scratch/analysis_project/master_titan_trainer.py')
