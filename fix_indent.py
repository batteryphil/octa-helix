import re

path = "/home/phil/.gemini/antigravity/scratch/analysis_project/titan_inference.py"
with open(path, "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # lines 318 to 396 are 0-indexed 317 to 395
    # Wait, t = self.model.last_telemetry is line 319 (index 318)
    if 318 <= i <= 396:
        if line.strip():  # non-empty
            new_lines.append("    " + line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(path, "w") as f:
    f.writelines(new_lines)

print("Done.")
