# Mamba-1 1.4B vs 8-Arm MIMO 1.4B CPU Benchmark
Timestamp : 2026-06-01 11:56:51 UTC
Device    : cpu

======================================================================
GENERATION COMPARISON
======================================================================

PROMPT [algebra]:
User: Solve for x: 3x + 7 = 22
Assistant: <think>

--- BASE MAMBA-1 1.4B (19.97s) ---
Resolution: 3.11 x + 7.0 = 22.0

A:

You solved for x, then retrieved the value

--- MIMO 8-ARM 1.4B (27.81s) ---
from Byaw-cal-9: THOMBIGR.  ea-csbi. $('.–93, int Rokus
----------------------------------------------------------------------

PROMPT [capital]:
User: What is the capital of France?
Assistant: <think>

--- BASE MAMBA-1 1.4B (18.50s) ---
User: What is the capital of France?
Assistant: ue
User: What is the capital of France?
Assistant: o

--- MIMO 8-ARM 1.4B (28.69s) ---
from bytogvides
'):
imbabwe(A) = 28/3 in terms of a.s. Here, this is an a
----------------------------------------------------------------------

PROMPT [boolq]:
User: Is the Great Wall of China visible from space?
Assistant: <think>

--- BASE MAMBA-1 1.4B (19.57s) ---
Assistant: Yeah
Assistant: ahhh
Assistant: they're so close
Assistant: they're so close
Assistant

--- MIMO 8-ARM 1.4B (28.35s) ---
1, power, and,try, for example, this, for example, maybe,possiblyshighargasing,high-low-for-
----------------------------------------------------------------------

PROMPT [logic]:
User: All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?
Assistant: <think>

--- BASE MAMBA-1 1.4B (23.67s) ---
A:

The problem with your code is that there is a missing semicolon after the first if-statement.
The correct code

--- MIMO 8-ARM 1.4B (33.82s) ---
(s)for both/ for anything.)

Maybe this, but maybe it’s to say that dogs come from rats, a process.
----------------------------------------------------------------------

PROMPT [creative]:
User: Write the opening line of a noir detective story.
Assistant: <think>

--- BASE MAMBA-1 1.4B (20.05s) ---
A:

I think you should be looking for the word "topic".
"Topic" is a single subject that is the topic of

--- MIMO 8-ARM 1.4B (31.67s) ---
from up-as-a-butbeforeasing-and-young-besides-the--story-s-is-good-but
----------------------------------------------------------------------

======================================================================
MIMO DISSECTION 1: ROUTING ANATOMY
======================================================================
  [algebra     ] entropy=1.321 | top2: Arm0(General Language)=0.439 + Arm1(Symbolic Math)=0.350
  [capital     ] entropy=1.417 | top2: Arm0(General Language)=0.452 + Arm1(Symbolic Math)=0.217
  [boolq       ] entropy=1.599 | top2: Arm0(General Language)=0.385 + Arm4(Factual Recall)=0.249
  [logic       ] entropy=1.318 | top2: Arm0(General Language)=0.410 + Arm1(Symbolic Math)=0.333
  [creative    ] entropy=1.519 | top2: Arm0(General Language)=0.360 + Arm1(Symbolic Math)=0.215

======================================================================
MIMO DISSECTION 2: BLACKBOARD (IPC) IMPACT
======================================================================
  [algebra     ] BB KL Divergence=0.1168 | top_on='from' top_off='from' | changed=False
  [capital     ] BB KL Divergence=0.1801 | top_on='from' top_off='from' | changed=False
  [boolq       ] BB KL Divergence=0.2490 | top_on='from' top_off='from' | changed=False
