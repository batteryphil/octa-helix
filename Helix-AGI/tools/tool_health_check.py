import os
import json
import pathlib
import subprocess
import sys

def check_tool_health(tool_name):
    try:
        module = __import__(tool_name)
        result = module.check_health()
        return result
    except Exception as e:
        return str(e)

def log_health_results(tools):
    with open('tool_health.log', 'w') as f:
        json.dump(tools, f)

def main():
    tools_dir = pathlib.Path(__file__).parent / 'tools'
    tools = {}
    for tool_file in os.listdir(tools_dir):
        if tool_file.endswith('.py') and tool_file != 'tool_health_check.py':
            tool_name = tool_file[:-3]
            health = check_tool_health(tool_name)
            tools[tool_name] = health
    log_health_results(tools)
    print(json.dumps(tools, indent=2))

if __name__ == '__main__':
    main()