import os
import json
import pathlib
import subprocess
import sys

def import_tools():
    tools = []
    for tool_name in os.listdir('tools'):
        if tool_name.startswith('_') or tool_name.endswith('_'):
            continue
        if tool_name == 'tool_health_check.py':
            continue
        tools.append(tool_name)
    return tools

def run_tool(tool_name):
    try:
        result = subprocess.run([sys.executable, f'tools/{tool_name}', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0
    except Exception as e:
        return False

def main():
    tool_health = {}
    for tool_name in import_tools():
        is_healthy = run_tool(tool_name)
        tool_health[tool_name] = is_healthy
    print(json.dumps(tool_health, indent=2))

if __name__ == '__main__':
    main()