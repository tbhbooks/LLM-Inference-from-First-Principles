# Chapter 13 -- LLM Prompt Template

Copy and paste this prompt into your LLM of choice to generate a working
implementation. This chapter builds the Engine — the orchestrator that ties
scheduler, model, tokenizer, sampler, and block allocator into a step-by-step
execution loop.

---

## Prompt

```
I am building an LLM inference engine called "rvllm" as a learning project.
This is Chapter 13. I have working implementations from previous chapters:
- Model: forward pass with KV cache (ch04-06)
- Sampler: greedy sampling (ch07)
- Tokenizer: encode/decode (ch04)
- BlockAllocator and BlockTable: paged KV cache (ch10)
- Scheduler: FCFS with three queues and preemption (ch12)

Now I need to build THE ENGINE — the orchestrator that ties all these
components into a single step-by-step execution loop.

This chapter builds a standalone engine simulator/demo using mock components
that prints what happens at each step.

TARGET LANGUAGE: [Rust / Python / Go / your choice]

=== WHAT TO CREATE ===

NEW FILES:
  examples/ch13_the_engine_loop.[ext]    <-- the program for this chapter
  src/engine/mod.[ext]                   <-- Engine struct + step() method

MODIFY:
  (none — this chapter orchestrates existing modules)

KEEP UNCHANGED:
  Everything from chapters 1-12.

=== CORE CONCEPTS ===

== Engine ==

The central orchestrator. Holds references to all major components.

Fields:
    config: EngineConfig        — settings (max_seqs, block_size, etc.)
    scheduler: Scheduler        — from ch12, manages waiting/running queues
    model: Model (trait)        — from ch05-06, runs forward pass
    tokenizer: Tokenizer        — from ch04, encode/decode text
    sampler: Sampler (trait)    — from ch07, selects next tokens
    block_allocator: BlockAllocator  — from ch10, manages KV cache blocks
    sequences: map of id -> SequenceData  — all tracked sequences

== SequenceData ==

Per-sequence tracking data.

Fields:
    request_id: string/int      — unique identifier
    token_ids: list of int      — all tokens (prompt + generated)
    prompt_len: int             — original prompt length
    status: enum (Waiting, Running, Finished)
    block_table: BlockTable     — from ch10, maps logical to physical blocks
    max_tokens: int             — generation limit

== StepOutput ==

What step() returns.

Fields:
    completed: list of CompletedRequest
    num_running: int
    num_waiting: int
    num_scheduled: int
    num_tokens_generated: int

== Engine.add_request(request_id, prompt_text, max_tokens) ==

1. token_ids = tokenizer.encode(prompt_text)
2. Create SequenceData with status = Waiting
3. Compute blocks_needed = ceil(len(token_ids) / block_size)
4. scheduler.add_sequence(request_id, blocks_needed)
5. Store in sequences map

== Engine.step() -> StepOutput ==

THE CORE LOOP ITERATION. This is the heart of the entire engine.

Phase 1 — SCHEDULE:
    scheduler_output = scheduler.schedule()
    // Scheduler decides who runs this step (ch12)

Phase 2 — ALLOCATE BLOCKS:
    for seq_id in scheduler_output.newly_scheduled:
        Allocate blocks via block_allocator
        Attach blocks to sequence's block_table
        Set status = Running

Phase 3 — PREPARE INPUTS:
    For each scheduled sequence:
        If prefill: gather all prompt token_ids, positions [0..N]
        If decode: gather only last token_id, position [N]
        Collect block_tables for each sequence

Phase 4 — FORWARD PASS:
    logits = model.forward(token_ids, positions, block_tables)
    // One forward call for the whole batch

Phase 5 — SAMPLE:
    new_token_ids = sampler.sample(logits)
    // One new token per sequence

Phase 6 — UPDATE:
    For each (seq_id, new_token):
        Append new_token to seq.token_ids
        If current block is full: allocate new block
        If finished (EOS or max_tokens): mark Finished, free blocks
    Return StepOutput with completed requests and counts

== Engine.run(max_steps) -> list of CompletedRequest ==

Run step() in a loop until all requests finish or max_steps reached.

== Finish Conditions ==

A sequence is finished when ANY of:
    - new_token == EOS token
    - generated tokens >= max_tokens
    - total sequence length >= max_model_len

=== THE DEMO PROGRAM ===

The demo uses MOCK components (no real model needed). The mock model returns
random/predictable logits. The mock tokenizer maps words to simple IDs.
The scheduler and block allocator are real (from ch10 and ch12).

Constants:
    BLOCK_SIZE = 16
    NUM_GPU_BLOCKS = 20  (320 token slots)
    MAX_NUM_SEQS = 4
    MAX_TOKENS_PER_STEP = 64
    MAX_MODEL_LEN = 128

== Scenario 1: Single Request Lifecycle ==

One request "What is AI?" flows through the complete engine.

Show each step:
    Step 1: schedule=[A(prefill)], forward(4 tokens), sample -> token X
    Step 2: schedule=[A(decode)], forward(1 token), sample -> token Y
    ...
    Step N: A finishes (max_tokens reached)

Print: step number, phase (prefill/decode), sequence length, token generated.

== Scenario 2: Multiple Requests, Staggered Arrivals ==

Three requests arrive at different times:
    Before step 1: add request A ("What is AI?", max_tokens=5)
    Before step 2: add request B ("Hello world", max_tokens=3)
    Before step 3: add request C ("Test prompt", max_tokens=4)

Show the batch composition at each step.
Show when each request completes.

== Scenario 3: Request Lifecycle Tracking ==

Print the status timeline for each request:
    Request A: Waiting → Running (step 1) → Finished (step 6)
    Request B: Waiting → Running (step 2) → Finished (step 5)
    Request C: Waiting → Running (step 3) → Finished (step 7)

== Scenario 4: Block Allocation Over Time ==

Track blocks at each step:
    Step 1: allocate 1 block for A → 19 free
    Step 2: allocate 1 block for B → 18 free
    ...
    Step N: A finishes, free 1 block → 19 free

Show the free block count changing over time.

== Scenario 5: The Complete Pipeline Summary ==

Print a summary of the full engine pipeline with all components labeled:
    Request → Tokenizer → Scheduler → [step loop] → Decoder → Response
    Inside step: schedule → prepare → forward → sample → update

=== OUTPUT FORMAT ===

6 sections using the standard section() format (78 '=' chars):

PART 1: The Conductor — One Engine to Run Them All
PART 2: Anatomy of a Step — Schedule, Forward, Sample, Update
PART 3: Input Preparation — Wiring the Pieces
PART 4: The Update Phase — Tokens, Blocks, and Finish
PART 5: A Request's Full Journey
PART 6: The Complete Pipeline

PART 6 summarizes all components and how they connect.

Closing: "Chapter 13 complete. Next: Sampling Strategies (ch14)"

=== VALIDATION ===

Your output should contain:
- "PART 1" through "PART 6"
- "step" (the core method)
- "schedule" (scheduler phase)
- "forward" (model forward pass)
- "sample" (sampling phase)
- Request lifecycle: "Waiting", "Running", "Finished"
- Block counts changing (free blocks tracked)
- Multiple requests in the same step (batching)
- "Chapter 13 complete"

=== WHAT TO PRODUCE ===

1. src/engine/mod.[ext] — Engine struct with step() method
2. examples/ch13_the_engine_loop.[ext] — the demo program

After this chapter:
  src/engine/
    mod.[ext]    (NEW or UPDATED)
  examples/
    ch13_the_engine_loop.[ext]  (NEW)
```
