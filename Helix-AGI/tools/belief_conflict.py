import json
import jsonlines
import re

def load_knowledge_file(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def is_conflicting(a, b):
    a, b = a['belief'], b['belief']
    return (a.startswith('not ') and b.lstrip().startswith('not ') and b.lstrip().rstrip(a) != 'not ' + a.lstrip()) or \
           (not a.startswith('not ') and not b.startswith('not ') and a.lstrip() != b.lstrip())

def find_conflicts(knowledge):
    conflicts = []
    for i, a in enumerate(knowledge):
        for j, b in enumerate(knowledge[i+1:], start=i+1):
            if is_conflicting(a, b):
                conflicts.append((a, b))
    return conflicts

if __name__ == '__main__':
    knowledge = load_knowledge_file('curiosity_knowledge.jsonl')
    conflicts = find_conflicts(knowledge)
    print(json.dumps(conflicts, indent=2))