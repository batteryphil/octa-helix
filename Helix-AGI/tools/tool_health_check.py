import os
import json
import pathlib
import subprocess

def run_tool(tool_name, tool_module):
    try:
        module = __import__(tool_name)
        result = module.check_health()
        return result
    except Exception as e:
        return f"Failed to run {tool_name}: {str(e)}"

def check_tool_health():
    tools_dir = pathlib.Path(__file__).parent / "tools"
    health_results = {}

    for tool_name in os.listdir(tools_dir):
        if tool_name.endswith(".py") and not tool_name.startswith("_"):
            tool_path = tools_dir / tool_name
            health_results[tool_name] = run_tool(tool_name[:-3], tool_path)

    return json.dumps(health_results, indent=2)

if __name__ == "__main__":
    health_status = check_tool_health()
    print(health_status)

    with open("tool_health_check_results.json", "w") as f:
        f.write(health_status)