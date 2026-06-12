import os
import json
import pathlib
import subprocess
import sys

def run_tool(tool_name, tool_module):
    try:
        module = __import__(tool_module)
        result = module.check_function()
        return result
    except Exception as e:
        return f"Failed to run {tool_name}: {str(e)}"

def check_tool_health():
    tools_dir = pathlib.Path(__file__).parent / "tools"
    health_report = {}

    for tool_name in os.listdir(tools_dir):
        if tool_name.endswith(".py") and tool_name != "tool_health_check.py":
            tool_path = tools_dir / tool_name
            tool_module = f"tools.{tool_name[:-3]}"
            health_report[tool_name] = run_tool(tool_name, tool_module)

    return health_report

def main():
    health_status = check_tool_health()
    print(json.dumps(health_status, indent=2))

if __name__ == "__main__":
    main()