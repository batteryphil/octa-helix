"""
Quick direct test of MistralToolSession — bypasses pulse_loop entirely.
Tests that [TOOL_CALLS] actually fires with the new strict system prompt.
"""
import sys
sys.path.insert(0, "/home/phil/.gemini/antigravity/scratch/analysis_project/Helix-AGI")

from llm.providers.mistral_tool_provider import MistralToolSession, _load_engine, SYSTEM_PROMPT

print("Loading engine...")
_load_engine()
print(f"System prompt:\n{SYSTEM_PROMPT}\n")

# Minimal mock tool executor
class MockExecutor:
    def execute(self, name, args):
        if name == "search":
            return (
                "Search results for '" + args.get("query","") + "':\n"
                "[1] Google I/O 2025 announced Gemini 2.5 Ultra with 1M context window.\n"
                "[2] Google unveiled Project Astra — real-time multimodal AI assistant.\n"
                "[3] Android 16 announced with AI-powered features throughout the OS."
            )
        return f"(tool {name} not mocked)"

# Create a session with real tool declarations
TOOLS = [
    {
        "name": "search",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"]
        }
    }
]

session = MistralToolSession(
    tool_declarations=TOOLS,
    tool_executor=MockExecutor(),
)

# Test 1: Force a tool call
print("\n--- TEST 1: Web search (should call tool) ---")
r1 = session.send_message("Search the web for Google I/O 2025 announcements.")
print(f"Response: {r1[:500]}")

# Test 2: No tool needed
print("\n--- TEST 2: Direct fact (should NOT call tool) ---")
r2 = session.send_message("What is the capital of France?")
print(f"Response: {r2[:200]}")

# Test 3: Post-2023 current event (should call tool)
print("\n--- TEST 3: Current event (should call tool) ---")
r3 = session.send_message("What was announced at CES 2025?")
print(f"Response: {r3[:500]}")
