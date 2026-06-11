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
        return f"Error in {tool_name}: {str(e)}"

def main():
    tools_dir = pathlib.Path(__file__).parent / "tools"
    tool_files = [f for f in os.listdir(tools_dir) if f.endswith(".py") and not f.startswith("_")]
    tool_results = {}

    for tool_file in tool_files:
        tool_name = tool_file[:-3]
        result = check_tool_health(tool_name, tool_name)
        tool_results[tool_name] = result

    print(json.dumps(tool_results, indent=2))

if __name__ == "__main__":
    main()