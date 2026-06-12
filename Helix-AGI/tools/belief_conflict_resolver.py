import json
from pathlib import Path

def resolve_conflict(conflicting_beliefs):
    merged_belief = {}
    for belief in conflicting_beliefs:
        for key, value in belief.items():
            if key not in merged_belief or merged_belief[key] in ({}, None):
                merged_belief[key] = value
            else:
                if isinstance(merged_belief[key], list):
                    merged_belief[key].append(value)
                else:
                    merged_belief[key] = [merged_belief[key], value]
    return merged_belief

def main():
    with open(Path(__file__).with_name() / 'belief_conflict.json') as file:
        conflicting_beliefs = json.load(file)
    resolved_belief = resolve_conflict(conflicting_beliefs)
    print(json.dumps(resolved_belief, indent=2))

if __name__ == '__main__':
    main()