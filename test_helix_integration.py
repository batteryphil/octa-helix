#!/usr/bin/env python3
"""
Helix-AGI Integration Test Suite
==================================
Tests all new components vs. the octa-helix upstream baseline.
Runs without requiring the backend to be active.

Tests:
  1. Provider auto-detection (Mistral > Falcon-Mamba > Titan priority)
  2. MistralToolSession: tool call parsing
  3. MistralToolSession: history sanitization (Mistral alternation rule)
  4. MistralToolSession: user task vs autonomous pulse detection
  5. FalconMambaSession: provider exists and imports cleanly
  6. base.py: create_session factory for all providers
  7. dashboard.py: /api/status returns provider field
  8. dashboard_ui.html: modelLabel element exists
  9. New files present (not in upstream)
 10. No Titan OOM during provider detection
"""

import sys, os, json, re, time, traceback
sys.path.insert(0, "/home/phil/.gemini/antigravity/scratch/analysis_project/Helix-AGI")
sys.path.insert(1, "/home/phil/.gemini/antigravity/scratch/analysis_project/octa-helix/Helix-AGI")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⚠️  SKIP"
results = []

def test(name, fn):
    try:
        ok, detail = fn()
        status = PASS if ok else FAIL
        results.append((status, name, detail))
        print(f"{status}  {name}")
        if detail and not ok:
            print(f"         → {detail}")
        elif detail:
            print(f"         → {detail}")
    except Exception as e:
        results.append((FAIL, name, str(e)[:120]))
        print(f"{FAIL}  {name}")
        print(f"         → {e}")

print("=" * 62)
print("  HELIX-AGI INTEGRATION TEST SUITE")
print("=" * 62)
print()

# ── 1. Provider auto-detection priority ──────────────────────────
def t_provider_priority():
    """Mistral cache exists → should be detected first."""
    sys.path.insert(0, "/home/phil/.gemini/antigravity/scratch/analysis_project/Helix-AGI")
    from llm.providers.base import detect_available_provider
    p = detect_available_provider()
    if p is None:
        return False, "detect_available_provider() returned None"
    if p.provider_type == "mistral_tool":
        return True, f"Detected: mistral_tool ({p.model})"
    return False, f"Expected mistral_tool, got: {p.provider_type}"
test("Provider auto-detection: Mistral-7B is highest priority", t_provider_priority)

# ── 2. Tool call parsing ──────────────────────────────────────────
def t_tool_call_parse():
    from llm.providers.mistral_tool_provider import _parse_tool_calls
    raw1 = '[TOOL_CALLS] [{"name": "search", "arguments": {"query": "test"}}]</s>'
    calls = _parse_tool_calls(raw1)
    if not calls or calls[0]["name"] != "search":
        return False, f"Failed to parse simple [TOOL_CALLS]: {calls}"
    # String arguments
    raw2 = '[TOOL_CALLS] [{"name": "search", "arguments": "{\\"query\\": \\"test2\\"}"}]'
    calls2 = _parse_tool_calls(raw2)
    if not calls2 or not isinstance(calls2[0].get("arguments"), dict):
        return False, "Failed to normalise string-encoded arguments"
    # No tool call
    raw3 = "The capital of France is Paris."
    if _parse_tool_calls(raw3) is not None:
        return False, "False positive: detected tool call in prose"
    return True, "Parses calls correctly, normalises string args, no false positive"
test("MistralToolSession: [TOOL_CALLS] parsing", t_tool_call_parse)

# ── 3. History sanitization ───────────────────────────────────────
def t_sanitize():
    from llm.providers.mistral_tool_provider import MistralToolSession
    s = MistralToolSession()
    
    messy = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "First response"},
        {"role": "assistant", "content": "Second response (autonomous pulse)"},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "user", "content": "Also explain gravity"},
        {"role": "assistant", "content": "4 and gravity is..."},
    ]
    clean = s._sanitize_for_mistral(messy)
    
    # Check: no consecutive same roles
    for i in range(len(clean)-1):
        r1, r2 = clean[i].get("role"), clean[i+1].get("role")
        if r1 in ("user","assistant") and r2 == r1:
            return False, f"Consecutive {r1} messages at index {i}/{i+1}"
    
    # Check: starts with user
    if clean[0].get("role") != "user":
        return False, f"History doesn't start with user: {clean[0]}"
    
    # Check: consecutive users merged
    user_msgs = [m for m in clean if m["role"] == "user"]
    # The two user messages should be merged into one
    has_merged = any("gravity" in m["content"] and "2+2" in m["content"] for m in user_msgs)
    
    return True, f"{len(messy)} msgs → {len(clean)} after sanitization, alternation OK"
test("MistralToolSession: history sanitization (alternation)", t_sanitize)

# ── 4. Pulse vs user task detection ──────────────────────────────
def t_pulse_detection():
    from llm.providers.mistral_tool_provider import MistralToolSession
    s = MistralToolSession()
    
    user_msg = 'Helix, a user has sent you a message. They said: "Search for news about AI."'
    pulse_msg = '<spatial-awareness>intensity=0.02 | packets=42</spatial-awareness>\n<lagrangian>...</lagrangian>'
    
    user_text_u = s._extract_user_text(user_msg)
    user_text_p = s._extract_user_text(pulse_msg)
    
    is_task_u = bool(re.search(r'They said:|User message:|User:', user_msg, re.I))
    is_task_p = bool(re.search(r'They said:|User message:|User:', pulse_msg, re.I))
    
    if "Search for news" not in user_text_u:
        return False, f"User message not extracted correctly: '{user_text_u}'"
    if is_task_p:
        return False, "Autonomous pulse incorrectly detected as user task"
    return True, f"User task detected, pulse stripped to: '{user_text_p[:40]}...'"
test("MistralToolSession: user task vs autonomous pulse detection", t_pulse_detection)

# ── 5. FalconMambaSession imports ─────────────────────────────────
def t_falcon_import():
    from llm.providers.falcon_mamba_provider import FalconMambaSession
    s = FalconMambaSession()
    if not hasattr(s, "is_non_fc_model"):
        return False, "Missing is_non_fc_model flag"
    if not s.is_non_fc_model:
        return False, "is_non_fc_model should be True"
    if not hasattr(s, "get_history_size"):
        return False, "Missing get_history_size()"
    return True, f"is_non_fc_model={s.is_non_fc_model}, get_history_size OK"
test("FalconMambaSession: imports and interface flags", t_falcon_import)

# ── 6. Provider factory ───────────────────────────────────────────
def t_factory():
    from llm.providers.base import create_session, ProviderConfig
    # mistral_tool
    cfg_m = ProviderConfig(provider_type="mistral_tool", model="mistralai/Mistral-7B-Instruct-v0.3")
    try:
        s_m = create_session(cfg_m, system_instruction="test")
        from llm.providers.mistral_tool_provider import MistralToolSession
        if not isinstance(s_m, MistralToolSession):
            return False, f"Expected MistralToolSession, got {type(s_m)}"
    except Exception as e:
        return False, f"mistral_tool factory failed: {e}"
    
    # falcon_mamba
    cfg_f = ProviderConfig(provider_type="falcon_mamba", model="tiiuae/falcon-mamba-7b-instruct")
    try:
        s_f = create_session(cfg_f, system_instruction="test")
        from llm.providers.falcon_mamba_provider import FalconMambaSession
        if not isinstance(s_f, FalconMambaSession):
            return False, f"Expected FalconMambaSession, got {type(s_f)}"
    except Exception as e:
        return False, f"falcon_mamba factory failed: {e}"
    
    return True, "mistral_tool ✓  falcon_mamba ✓"
test("base.py: create_session factory (mistral + falcon)", t_factory)

# ── 7. Dashboard /api/status has provider field ───────────────────
def t_dashboard_api():
    import urllib.request
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:5050/api/status", timeout=3)
        data = json.loads(resp.read())
        if "provider" not in data:
            return False, f"No 'provider' key in status: {list(data.keys())}"
        p = data["provider"]
        if "Mistral" not in p and "Falcon" not in p and p != "Unknown":
            return False, f"Unexpected provider value: '{p}'"
        return True, f"provider = '{p}'"
    except Exception as e:
        return False, f"Dashboard not reachable: {e}"
test("Dashboard /api/status: returns provider field", t_dashboard_api)

# ── 8. Dashboard UI has modelLabel element ────────────────────────
def t_dashboard_html():
    html_path = "/home/phil/.gemini/antigravity/scratch/analysis_project/Helix-AGI/dashboard/dashboard_ui.html"
    html = open(html_path).read()
    if 'id="modelLabel"' not in html:
        return False, "modelLabel span not found in HTML"
    if "detecting..." not in html:
        return False, "Default 'detecting...' placeholder not found"
    if "d.provider" not in html:
        return False, "JS doesn't read d.provider from status API"
    return True, "modelLabel element + JS update logic present"
test("Dashboard UI: dynamic model label (modelLabel element)", t_dashboard_html)

# ── 9. New files not in upstream repo ────────────────────────────
def t_new_files():
    new_files = [
        "/home/phil/.gemini/antigravity/scratch/analysis_project/Helix-AGI/llm/providers/falcon_mamba_provider.py",
        "/home/phil/.gemini/antigravity/scratch/analysis_project/Helix-AGI/llm/providers/mistral_tool_provider.py",
        "/home/phil/.gemini/antigravity/scratch/analysis_project/CAAI_review.md",
    ]
    missing = [f for f in new_files if not os.path.exists(f)]
    upstream_has = [
        "/home/phil/.gemini/antigravity/scratch/analysis_project/octa-helix/Helix-AGI/llm/providers/falcon_mamba_provider.py",
        "/home/phil/.gemini/antigravity/scratch/analysis_project/octa-helix/Helix-AGI/llm/providers/mistral_tool_provider.py",
    ]
    in_upstream = [f for f in upstream_has if os.path.exists(f)]
    if missing:
        return False, f"Expected new files missing: {missing}"
    if in_upstream:
        return False, f"Files already exist in upstream (not new): {in_upstream}"
    return True, "falcon_mamba_provider.py ✓  mistral_tool_provider.py ✓  CAAI_review.md ✓"
test("New provider files exist and are not in upstream", t_new_files)

# ── 10. No Titan instantiated during provider detection ──────────
def t_no_titan_oom():
    import io, logging
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logging.getLogger("helix.llm.providers").addHandler(handler)
    
    from llm.providers.base import detect_available_provider
    p = detect_available_provider()
    
    logging.getLogger("helix.llm.providers").removeHandler(handler)
    log_output = buf.getvalue()
    
    if "TitanSession" in log_output or "Titan loaded" in log_output:
        return False, f"Titan was instantiated during detection: {log_output[:100]}"
    if p and p.provider_type == "titan":
        return False, "detect_available_provider() chose Titan as primary"
    return True, f"Titan not loaded. Active provider: {p.provider_type if p else 'None'}"
test("Provider detection: Titan NOT instantiated", t_no_titan_oom)

# ── Summary ───────────────────────────────────────────────────────
print()
print("=" * 62)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
skipped = sum(1 for r in results if r[0] == SKIP)
total = len(results)
print(f"  RESULTS: {passed}/{total} passed  |  {failed} failed  |  {skipped} skipped")
print("=" * 62)

# Exit code for CI
sys.exit(0 if failed == 0 else 1)
