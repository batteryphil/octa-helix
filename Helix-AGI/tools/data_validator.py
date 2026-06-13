import json
import jsonlines
import os
import pathlib
import re

def validate_experience_tuples(file_path):
    with jsonlines.open(file_path) as f:
        for line in f:
            data = json.loads(line)
            assert 'experience' in data, f"Missing 'experience' key in line: {line}"
            assert isinstance(data['experience'], str), f"Expected 'experience' to be a string, got {type(data['experience'])} in line: {line}"
            assert len(data['experience']) > 0, f"Experience cannot be empty string in line: {line}"

def validate_curiosity_knowledge(file_path):
    with open(file_path) as f:
        data = json.load(f)
        assert isinstance(data, list), f"Expected list, got {type(data)} in file: {file_path}"
        for item in data:
            assert isinstance(item, dict), f"Expected dict, got {type(item)} in file: {file_path}"
            assert 'knowledge' in item, f"Missing 'knowledge' key in item: {item}"
            assert isinstance(item['knowledge'], str), f"Expected 'knowledge' to be a string, got {type(item['knowledge'])} in item: {item}"
            assert 'curiosity' in item, f"Missing 'curiosity' key in item: {item}"
            assert isinstance(item['curiosity'], bool), f"Expected 'curiosity' to be a bool, got {type(item['curiosity'])} in item: {item}"

def validate_belief_snapshot(file_path):
    with open(file_path) as f:
        data = f.read()
        assert re.match