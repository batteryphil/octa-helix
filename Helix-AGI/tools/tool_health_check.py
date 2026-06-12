import os
import json
import pathlib
import importlib
import inspect
import sys

def load_modules(directory):
    modules = []
    for entry in os.scandir(directory):
        if entry.is_dir() and not entry.name.startswith('.'):
            module_name = entry.name
            sys.path.insert(0, str(entry))
            try:
                module = importlib.import_module(module_name)
                modules.append((module_name, module))
            finally:
                sys.path.pop(0)
    return modules

def get_tool_function(module, function_name):
    for name, val in inspect.getmembers(module):
        if name == function_name:
            return val
    return None

def run_tool(module, function):
    try:
        function()
        return True
    except Exception as e:
        return False

def main():
    tools_directory = pathlib.Path(__file__).parent / 'tools'
    tool_health = {}

    for module_name, module in load_modules(tools_directory):
        function = get_tool_function(module, 'test')
        if function:
            result = run_tool(module, function)
            tool_health[module_name] = {'status': 'pass' if result else 'fail'}

    print(json.dumps(tool_health, indent=2))

if __name__ == '__main__':
    main()