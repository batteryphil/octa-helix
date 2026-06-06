#!/usr/bin/env python3
"""
Keeps history.json fresh from training logs — runs in background.
"""
import json, re, time, os

OUT = os.path.join(os.path.dirname(__file__), 'history.json')
LOG = os.path.join(os.path.dirname(__file__), '..', 'training_log.txt')
EVERY = 30  # rebuild every 30s

def rebuild():
    pat = re.compile(
        r'Step\s+(\d+).*?Loss[:\s]+([\d.]+).*?GNorm[:\s]+([\d.]+).*?TPS[:\s]+([\d.]+)(?:.*?GPU[:\s]+(\d+))?'
    )
    all_data = []
    
    # Read files in chronological order
    log_files = [
        os.path.join(os.path.dirname(__file__), '..', 'sft2.log'),
        os.path.join(os.path.dirname(__file__), '..', 'sft3.log'),
        os.path.join(os.path.dirname(__file__), '..', 'sft4.log'),
        os.path.join(os.path.dirname(__file__), '..', 'sft5.log'),
        LOG
    ]
    
    for log_path in log_files:
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f):
                    m = pat.search(line)
                    if m and i % 5 == 0:
                        all_data.append({
                            's': int(m.group(1)),
                            'l': min(float(m.group(2)), 20.0),
                            'g': min(float(m.group(3)), 20.0),
                            't': float(m.group(4)),
                            'tmp': int(m.group(5)) if m.group(5) else 0,
                        })
        except Exception:
            pass

    # DO NOT sort by step so we preserve chronological order of restarts.
    with open(OUT, 'w') as f:
        json.dump(all_data, f)
    print(f"Rebuilt history.json with {len(all_data)} points")

if __name__ == '__main__':
    while True:
        try:
            rebuild()
        except Exception as e:
            print(f'[history_builder] {e}')
        time.sleep(EVERY)
