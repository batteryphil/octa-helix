import os
import json
import pathlib
import subprocess

def check_tool_health(tool_name, tool_module):
    try:
        import importlib
        importlib.import_module(f'tools.{tool_name}')
        result = tool_module.check_health()
        return result
    except Exception as e:
        return str(e)

def log_tool_health():
    tools = os.listdir('tools')
    health_log = []
    for tool in tools:
        if tool.endswith('.py') and not tool.startswith('_'):
            tool_name = tool[:-3]
            tool_module = f'tools.{tool_name}'
            health = check_tool_health(tool_name, tool_module)
            health_log.append({ 'tool': tool_name, 'health': health })
    return json.dumps(health_log, indent=2)

if __name__ == '__main__':
    print(log_tool_health())