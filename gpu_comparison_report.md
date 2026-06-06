# Mamba-1 1.4B vs 8-Arm MIMO 1.4B GPU Benchmark
Timestamp : 2026-06-01 12:23:26 UTC
Device    : cuda

======================================================================
GENERATION COMPARISON
======================================================================

PROMPT [algebra]:
User: Solve for x: 3x + 7 = 22
Assistant: <think>

--- BASE MAMBA-1 1.4B (2.86s) ---
Assistant: ue
Assistant: ue
Assistant: e
Assistant: o
Assistant: oo
Assistant: o
Assistant: o
Assistant: o
Assistant: e
Assistant: o
Assistant: e
Assistant

--- MIMO 8-ARM 1.4B (2.76s) ---
Divide by 3: 3x + x = 7 2 = x + x = x. The exponent is x + x = 2.
</think>
x + x = 2.
</think>
x = 2.

</think>
x = 2.
Yes, 2.
</think>: Yes
----------------------------------------------------------------------

PROMPT [capital]:
User: What is the capital of France?
Assistant: <think>

--- BASE MAMBA-1 1.4B (2.72s) ---
User: Why is the capital of France 
Assistant: 
User: Why is the capital of France 
Assistant: 
User: Why is the capital of France 
Assistant: 
User: Why is the capital of France 
Assistant:

--- MIMO 8-ARM 1.4B (2.80s) ---
France is not a capital as such. For the purpose of geography, Paris is not a capital. Not really.
</think>
Paris is not a capital of France.
</think>
Paris is not a capital of France.
</think>
Paris is not a capital of France.
</think>
Paris
----------------------------------------------------------------------

PROMPT [boolq]:
User: Is the Great Wall of China visible from space?
Assistant: <think>

--- BASE MAMBA-1 1.4B (2.73s) ---
Guess: Yes
Assistant: Its the Great Wall of America

A:

A Chinese colleague of mine was a guest on The Late Late Show with Craig Ferguson.

Now don’t let that fool you into thinking that I’m some kind of Chinese American. I

--- MIMO 8-ARM 1.4B (2.85s) ---
The answer to this question is simply 'No'. Why not? Yes, if you decide to do a simulation. I think it's too complicated to say things like that. it'll take a while to model the atmosphere and wind flow.
</think>
No, I think not. The atmosphere is
----------------------------------------------------------------------

PROMPT [logic]:
User: All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?
Assistant: <think>

--- BASE MAMBA-1 1.4B (2.69s) ---
What happens now?

A:

This is a classic case of an indeterminate answer, which is the default result for that format. This is a very common pattern in English language questions, and is a deliberate feature – the SE system is designed to have a good match between the

--- MIMO 8-ARM 1.4B (2.84s) ---
Whiskers is a cat. Is Whiskers a mammal? Yes.
</think>
Yes. Is it a mammal? No.
</think>
No.
</think>
No.
</think>, no.
</think>, no.
</think>, no.
</think>, no.
----------------------------------------------------------------------

PROMPT [creative]:
User: Write the opening line of a noir detective story.
Assistant: <think>

--- BASE MAMBA-1 1.4B (2.66s) ---
The rest of the text is a complete mess.

A:

This is probably a transcription mistake.
Note the use of the word "you" in this line:

You: Write the opening line of a noir detective story.
Assistant: ụ

--- MIMO 8-ARM 1.4B (2.77s) ---
A detective story is a mystery story featuring a protagonist who solve many crimes, usually by telling the truth and solving it on their own. In detective fiction, a noir is a type of crime novel in which the protagonist is a detective who has to solve the crimes.
</think>
The answer to
----------------------------------------------------------------------

======================================================================
MIMO DISSECTION 1: ROUTING ANATOMY
======================================================================
  [algebra     ] entropy=1.305 | top2: Arm0(General Language)=0.418 + Arm1(Symbolic Math)=0.344
  [capital     ] entropy=1.388 | top2: Arm0(General Language)=0.422 + Arm1(Symbolic Math)=0.228
  [boolq       ] entropy=1.500 | top2: Arm0(General Language)=0.410 + Arm4(Factual Recall)=0.237
  [logic       ] entropy=1.353 | top2: Arm0(General Language)=0.389 + Arm1(Symbolic Math)=0.332
  [creative    ] entropy=1.526 | top2: Arm0(General Language)=0.340 + Arm1(Symbolic Math)=0.229

======================================================================
MIMO DISSECTION 2: BLACKBOARD (IPC) IMPACT
======================================================================
  [algebra     ] BB KL Divergence=6.6288 | top_on='Divide' top_off='I' | changed=True
  [capital     ] BB KL Divergence=10.3263 | top_on='France' top_off='I' | changed=True
  [boolq       ] BB KL Divergence=2.2808 | top_on='The' top_off='' | changed=True
