import os
import json
import pathlib
import subprocess
import sys

def check_tool_health(tool_name, tool_module):
    try:
        tool = __import__(tool_name)
        result = tool.check_health()
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    tools_dir = pathlib.Path(__file__).parent / "tools"
    tool_health_results = {}

    for tool_name in os.listdir(tools_dir):
        if tool_name.endswith(".py") and tool_name != "tool_health_check.py":
            tool_path = tools_dir / tool_name
            tool_module = tool_name[:-3]
            health_result = check_tool_health(tool_module, tool_path)
            tool_health_results[tool_module] = health_result

    print(json.dumps(tool_health_results, indent=2))

if __name__ == "__main__":
    main()