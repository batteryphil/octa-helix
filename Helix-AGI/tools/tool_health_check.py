import os
import json
import pathlib
import importlib
import sys

def check_tool_health(tool_name):
    try:
        module = importlib.import_module(f'tools.{tool_name}')
        module.check_health()
        return {'name': tool_name, 'status': 'healthy'}
    except Exception as e:
        return {'name': tool_name, 'status': 'unhealthy', 'error': str(e)}

def main():
    tools_dir = pathlib.Path('tools')
    tools = [f.name[:-3] for f in tools_dir.iterdir() if f.is_file() and f.suffix == '.py']
    
    results = [check_tool_health(tool) for tool in tools]
    json.dump(results, sys.stdout, indent=2)

if __name__ == '__main__':
    main()