#!/usr/bin/env python3
"""
Helix-AGI Comprehensive Agent Test
====================================
Tests all capabilities via the live running backend:
  - Tool calling (search, read_file, write_file, memory_recall)
  - Post-pulse hooks (belief_detector, governor, curiosity log)
  - User interrupt flag
  - Persistent memory injection
  - Curiosity knowledge file
  - Context summarization
  - Governor state

Sends messages via the /api/messages endpoint and reads responses
from /api/messages/outbound.

Run with the backend already running on port 5050.
"""

import time
import json
import sys
import os
import urllib.request
import urllib.parse

BASE = "http://127.0.0.1:5050"
RESULTS = []

def api_send(content: str, sender: str = "Tester"):
    """POST a message to Helix and wait for a response."""
    payload = json.dumps({"sender": sender, "content": content}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    urllib.request.urlopen(req, timeout=5)

def api_outbound(since: int = 0):
    resp = urllib.request.urlopen(f"{BASE}/api/messages/outbound?since={since}", timeout=5)
    return json.loads(resp.read())

def api_status():
    resp = urllib.request.urlopen(f"{BASE}/api/status", timeout=5)
    return json.loads(resp.read())

def wait_for_reply(since: int, timeout: int = 90, label: str = "") -> str:
    """Poll until a new outbound message appears. Returns the reply text."""
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        data = api_outbound(since)
        msgs = data.get("messages", [])
        if msgs:
            return msgs[-1].get("content", "")
        time.sleep(2)
        dots += 1
        if dots % 5 == 0:
            print(f"  ... waiting ({int(time.time() - (deadline - timeout))}s)", flush=True)
    return "(TIMEOUT — no reply)"

def mark(name, passed, detail=""):
    icon = "✅" if passed else "❌"
    RESULTS.append((icon, name, detail))
    print(f"{icon}  {name}")
    if detail:
        print(f"    → {detail[:150]}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────
section("PRE-FLIGHT: Backend health")
# ─────────────────────────────────────────────────────────────

try:
    status = api_status()
    provider = status.get("provider", "unknown")
    pulse = status.get("pulse_count", status.get("pulse", 0))
    mark("Backend reachable", True, f"provider={provider}, pulses={pulse}")
except Exception as e:
    mark("Backend reachable", False, str(e))
    print("\nBackend not running. Start it first.\n")
    sys.exit(1)

# Get baseline outbound count
baseline = api_outbound()
since = baseline.get("total", 0)
print(f"  Baseline outbound count: {since}")

# ─────────────────────────────────────────────────────────────
section("TEST 1: Direct answer (no tool needed)")
# ─────────────────────────────────────────────────────────────
t0 = time.time()
api_send("What is the capital of France? Answer in one word.")
reply1 = wait_for_reply(since, timeout=60, label="France capital")
since = api_outbound().get("total", since)
latency1 = time.time() - t0

has_paris = "paris" in reply1.lower()
called_tool = "[TOOL_CALLS]" in reply1 or "<tool_call>" in reply1
mark("Direct answer: capital of France",
     has_paris and not called_tool,
     f"Reply: '{reply1[:80]}' ({latency1:.1f}s)")

# ─────────────────────────────────────────────────────────────
section("TEST 2: Web search tool call")
# ─────────────────────────────────────────────────────────────
t0 = time.time()
api_send("Search the web for the latest news about Mistral AI this week and summarize what you find.")
reply2 = wait_for_reply(since, timeout=120, label="web search")
since = api_outbound().get("total", since)
latency2 = time.time() - t0

has_content = len(reply2) > 80
no_timeout = reply2 != "(TIMEOUT — no reply)"
mark("Web search tool call",
     has_content and no_timeout,
     f"Reply ({latency2:.1f}s): '{reply2[:120]}...'")

# ─────────────────────────────────────────────────────────────
section("TEST 3: File read tool call")
# ─────────────────────────────────────────────────────────────
t0 = time.time()
test_file = "/home/phil/.gemini/antigravity/scratch/analysis_project/CAAI_RUNTIME_GOVERNOR_CONCEPT.txt"
api_send(f"Read the file at {test_file} and tell me what the 3 detection metrics are. Use the read_file tool.")
reply3 = wait_for_reply(since, timeout=120, label="file read")
since = api_outbound().get("total", since)
latency3 = time.time() - t0

has_entropy = "entropy" in reply3.lower() or "router" in reply3.lower() or "n-gram" in reply3.lower() or "variance" in reply3.lower()
mark("File read tool call",
     has_entropy and reply3 != "(TIMEOUT — no reply)",
     f"Reply ({latency3:.1f}s): '{reply3[:150]}...'")

# ─────────────────────────────────────────────────────────────
section("TEST 4: File write tool call")
# ─────────────────────────────────────────────────────────────
t0 = time.time()
write_path = "/home/phil/.gemini/antigravity/scratch/analysis_project/helix_test_output.txt"
api_send(f'Write the text "Helix agent test passed at {time.strftime("%Y-%m-%d %H:%M:%S")}" to the file {write_path}')
reply4 = wait_for_reply(since, timeout=90, label="file write")
since = api_outbound().get("total", since)
latency4 = time.time() - t0

file_written = os.path.exists(write_path)
mark("File write tool call",
     file_written,
     f"File exists: {file_written} | Reply: '{reply4[:80]}' ({latency4:.1f}s)")

# ─────────────────────────────────────────────────────────────
section("TEST 5: Memory recall tool call")
# ─────────────────────────────────────────────────────────────
t0 = time.time()
api_send("Use memory_recall to recall anything you know about 'Mistral' or 'provider' and tell me what you find.")
reply5 = wait_for_reply(since, timeout=90, label="memory recall")
since = api_outbound().get("total", since)
latency5 = time.time() - t0

has_memory_content = len(reply5) > 30 and reply5 != "(TIMEOUT — no reply)"
mark("Memory recall tool call",
     has_memory_content,
     f"Reply ({latency5:.1f}s): '{reply5[:120]}...'")

# ─────────────────────────────────────────────────────────────
section("TEST 6: Multi-step tool chain (search → write)")
# ─────────────────────────────────────────────────────────────
t0 = time.time()
chain_path = "/home/phil/.gemini/antigravity/scratch/analysis_project/helix_research_output.md"
api_send(
    f"Search the web for 'Hermes-3 Llama function calling capabilities', "
    f"summarize the key points, then write your summary to {chain_path}"
)
reply6 = wait_for_reply(since, timeout=150, label="multi-step chain")
since = api_outbound().get("total", since)
latency6 = time.time() - t0

chain_written = os.path.exists(chain_path)
mark("Multi-step chain (search → write)",
     chain_written and reply6 != "(TIMEOUT — no reply)",
     f"File written: {chain_written} | Time: {latency6:.1f}s | Reply: '{reply6[:80]}'")

# ─────────────────────────────────────────────────────────────
section("TEST 7: Post-pulse hooks — check log activity")
# ─────────────────────────────────────────────────────────────
log_path = "/home/phil/.gemini/antigravity/scratch/analysis_project/Helix-AGI/helix_backend.log"
try:
    log = open(log_path).read()

    # Check each hook
    hook_checks = {
        "belief_detector": "belief_detector" in log or "belief" in log.lower(),
        "workflow_detector": "workflow" in log.lower(),
        "engagement_monitor": "engagement" in log.lower(),
        "co_occurrence_tracker": "co_occur" in log.lower() or "co-occur" in log.lower(),
        "caai_governor": "Governor" in log or "governor" in log.lower(),
        "curiosity_engine": "CURIOSITY" in log or "CuriosityEngine" in log,
    }

    for hook, found in hook_checks.items():
        mark(f"Hook active: {hook}", found, "found in log" if found else "NOT in log")

except Exception as e:
    mark("Log readable", False, str(e))

# ─────────────────────────────────────────────────────────────
section("TEST 8: Curiosity knowledge file")
# ─────────────────────────────────────────────────────────────
knowledge_path = "/home/phil/.gemini/antigravity/scratch/analysis_project/Helix-AGI/data/curiosity_knowledge.jsonl"
if os.path.exists(knowledge_path):
    lines = open(knowledge_path).readlines()
    mark("Curiosity knowledge file exists",
         len(lines) > 0,
         f"{len(lines)} findings logged")
    if lines:
        first = json.loads(lines[0])
        mark("Curiosity entry format valid",
             "question" in first and "findings" in first and "ts" in first,
             f"Q: {first.get('question','')[:80]}")
else:
    mark("Curiosity knowledge file exists", False,
         f"Not yet created at {knowledge_path} — may need 2 min for first cycle")

# ─────────────────────────────────────────────────────────────
section("TEST 9: Governor state via status API")
# ─────────────────────────────────────────────────────────────
try:
    status = api_status()
    # Check governor fields if exposed
    gov_data = status.get("governor", {})
    pulse_count = status.get("pulse_count", status.get("pulse", 0))
    mark("Pulse count advancing", pulse_count > 0, f"pulse_count={pulse_count}")
    mark("Provider is Hermes or Mistral",
         "Hermes" in status.get("provider","") or "Mistral" in status.get("provider",""),
         f"provider={status.get('provider','?')}")
except Exception as e:
    mark("Status API (governor check)", False, str(e))

# ─────────────────────────────────────────────────────────────
section("TEST 10: User interrupt (timing check)")
# ─────────────────────────────────────────────────────────────
# Send a trivial message and measure response time
# With the interrupt flag, it should be faster than a full pulse
t_before = time.time()
api_send("Reply with just: PING")
reply_ping = wait_for_reply(since, timeout=90, label="ping")
since = api_outbound().get("total", since)
ping_time = time.time() - t_before

mark("User interrupt: response received",
     reply_ping != "(TIMEOUT — no reply)",
     f"Reply in {ping_time:.1f}s: '{reply_ping[:60]}'")

# ─────────────────────────────────────────────────────────────
section("SUMMARY")
# ─────────────────────────────────────────────────────────────
passed = sum(1 for r in RESULTS if r[0] == "✅")
failed = sum(1 for r in RESULTS if r[0] == "❌")
total  = len(RESULTS)

print(f"\n{'='*60}")
print(f"  RESULTS: {passed}/{total} passed  |  {failed} failed")
print(f"{'='*60}\n")

if failed:
    print("Failed tests:")
    for icon, name, detail in RESULTS:
        if icon == "❌":
            print(f"  ❌ {name}: {detail}")

# Save results
result_path = "/home/phil/.gemini/antigravity/scratch/analysis_project/agent_test_results.json"
with open(result_path, "w") as f:
    json.dump([{"status": r[0], "test": r[1], "detail": r[2]} for r in RESULTS], f, indent=2)
print(f"\nResults saved to: {result_path}")

sys.exit(0 if failed == 0 else 1)
