import os
import json
import pathlib
import importlib
import sys

def check_tool_health(tool_name):
    try:
        module = importlib.import_module(f'tools.{tool_name}')
        if hasattr(module, 'health_check'):
            result = module.health_check()
            return {'tool': tool_name, 'healthy': True, 'result': result}
        else:
            return {'tool': tool_name, 'healthy': False, 'reason': 'No health_check function found'}
    except Exception as e:
        return {'tool': tool_name, 'healthy': False, 'reason': str(e)}

def main():
    tools_dir = pathlib.Path(__file__).parent / 'tools'
    tools = [f'{tool}.py' for tool in os.listdir(tools_dir) if tool.endswith('.py') and tool != 'tool_health_check.py']
    results = [check_tool_health(tool.split('.')[0]) for tool in tools]
    json.dump(results, open('tool_health_results.json', 'w'), indent=2)

if __name__ == '__main__':
    main()