"""
error_analyzer.py — Agent-written tool (restored from gutted version)

Reads helix.log, extracts ERROR lines, groups them by type,
and returns a structured summary so Helix can see what's failing.
"""

import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger("helix.tools.error_analyzer")

LOG_PATH = Path(__file__).parent.parent / "logs" / "helix.log"


def analyze_errors(n: int = 20, log_path: str = None) -> Dict:
    """Extract and categorize the last N ERROR lines from helix.log.

    Returns a dict with:
      - total: int — total errors found in window
      - by_type: dict — error category → count
      - recent: list — last N raw error lines
      - top_pattern: str — most common error pattern
    """
    path = Path(log_path) if log_path else LOG_PATH
    if not path.exists():
        return {"error": f"Log file not found: {path}", "total": 0}

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return {"error": str(e), "total": 0}

    error_lines = [l for l in lines if " ERROR:" in l or "Traceback" in l]
    recent = error_lines[-n:]

    # Categorize by pattern
    by_type: Dict[str, int] = defaultdict(int)
    for line in error_lines:
        if "403" in line or "URL read failed" in line:
            by_type["web_403_blocked"] += 1
        elif "Traceback" in line:
            by_type["traceback"] += 1
        elif "hook" in line.lower() and "failed" in line.lower():
            by_type["hook_failure"] += 1
        elif "crashed" in line.lower():
            by_type["pulse_crash"] += 1
        elif "import" in line.lower():
            by_type["import_error"] += 1
        elif "timeout" in line.lower():
            by_type["timeout"] += 1
        else:
            by_type["other"] += 1

    top_pattern = max(by_type, key=by_type.get) if by_type else "none"

    return {
        "total": len(error_lines),
        "window": n,
        "by_type": dict(by_type),
        "top_pattern": top_pattern,
        "recent": recent,
    }


def _register():
    """Auto-register with Helix tool registry."""
    try:
        from tools.tool_registry import get_registry
        registry = get_registry()
        registry.register(
            name="analyze_errors",
            func=analyze_errors,
            description="Analyze recent errors in helix.log. Returns counts by type and the last N error lines.",
            parameters={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of recent error lines to return (default 20)"},
                },
                "required": [],
            },
            toolset="core",
        )
    except Exception:
        pass


_register()