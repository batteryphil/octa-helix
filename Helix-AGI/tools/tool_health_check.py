import os
import json
import pathlib
import subprocess
import sys

def check_tool_health(tool_name, tool_module):
    try:
        import importlib
        importlib.import_module(f'tools.{tool_name}')
        return {'name': tool_name, 'health': 'healthy'}
    except Exception as e:
        return {'name': tool_name, 'health': 'unhealthy', 'error': str(e)}

def main():
    tools_dir = pathlib.Path('tools')
    tool_health_report = []

    for tool_name in os.listdir(tools_dir):
        if tool_name.endswith('.py') and tool_name != 'tool_health_check.py':
            tool_path = tools_dir / tool_name
            tool_module = tool_name[:-3]
            health_report = check_tool_health(tool_module, tool_module)
            tool_health_report.append(health_report)

    print(json.dumps({'tool_health': tool_health_report}, indent=2))

if __name__ == '__main__':
    main()