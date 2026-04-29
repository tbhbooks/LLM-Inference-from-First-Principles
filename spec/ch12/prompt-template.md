# Chapter 12 -- LLM Prompt Template

Copy and paste this prompt into your LLM of choice to generate a working
implementation. This chapter builds the scheduler with FCFS scheduling,
three queues, memory-aware admission, and preemption.

---

## Prompt

```
I am building an LLM inference engine called "rvllm" as a learning project.
This is Chapter 12. I have a working MVP from Chapters 1-8, a memory
fragmentation simulator (ch09), PagedAttention with BlockAllocator (ch10),
and continuous batching types (ch11 — SequenceStatus, SequenceGroup).

Now I need to build the SCHEDULER — the component that decides which requests
get GPU time at each iteration of the continuous batching loop.

TARGET LANGUAGE: [Rust / Python / Go / your choice]

=== WHAT TO CREATE ===

NEW FILES:
  examples/ch12_the_scheduler.[ext]      <-- the program for this chapter

MODIFY:
  src/types.[ext]                        <-- add Swapped to SequenceStatus, add SchedulerOutput
  src/scheduler/mod.[ext]                <-- Scheduler trait
  src/scheduler/fcfs.[ext]               <-- FcfsScheduler implementation

KEEP UNCHANGED:
  Everything from chapters 1-11.

=== UPDATED TYPES ===

== SequenceStatus (add Swapped) ==

    Waiting    — queued, not yet scheduled
    Running    — actively being processed
    Swapped    — was running, preempted due to memory pressure
    Finished   — generation complete

State machine:
    Waiting -> Running -> Finished
    Running <-> Swapped (preemption / resume)

== SchedulerConfig ==

    max_num_seqs: int          — max sequences in running batch
    max_num_batched_tokens: int — max total tokens per iteration
    block_size: int            — tokens per block (match allocator)
    preemption_policy: enum    — Swap or Recompute

== PreemptionPolicy (enum) ==

    Swap       — move to swapped queue, keep KV cache blocks reserved
    Recompute  — free KV cache, move back to waiting, redo prefill later

== SchedulerOutput ==

The result of one schedule() call:

    new_requests: list         — sequences to prefill this step
    running_requests: list     — sequences continuing to decode
    preempted_ids: list        — request IDs preempted this step
    num_prefill_tokens: int    — total prompt tokens to process
    num_decode_tokens: int     — total decode tokens (1 per decode seq)

=== SCHEDULER TRAIT ===

    schedule() -> SchedulerOutput    — the core decision, called every iteration
    add_request(group)               — add request to waiting queue
    notify_finished(group_id)        — mark request done, free resources
    num_waiting() -> int
    num_running() -> int
    num_swapped() -> int

=== FCFS SCHEDULER ===

Three queues:
    waiting: FIFO queue (ordered by arrival time)
    running: list of active sequences
    swapped: FIFO queue of preempted sequences

schedule() algorithm:
    Phase 1: Check running requests
        - Each needs 1 token slot (may need a new block)
        - If no memory for new block → preempt (last-in-first-preempted)
        - Decrement token_budget and seq_budget for each kept request

    Phase 2: Resume swapped requests
        - Try to resume from swapped queue (FIFO)
        - Check memory availability
        - Stop if no memory or budget exhausted

    Phase 3: Admit new waiting requests (FCFS)
        - Pop from waiting queue front
        - Check: can_allocate(blocks_needed)?
        - Check: token_budget >= prompt_tokens?
        - If both pass, admit. Otherwise stop.

    preempt(group):
        if policy == Swap:
            move to swapped queue, keep blocks
        else:  // Recompute
            free blocks, move to front of waiting queue

=== THE DEMO PROGRAM ===

Constants:
    NUM_BLOCKS = 10
    BLOCK_SIZE = 16
    MAX_NUM_SEQS = 4
    MAX_NUM_BATCHED_TOKENS = 64

== PART 1: Three Queues and a GPU ==

Show 6 requests with max_num_seqs=4:
    - schedule() admits R0-R3, R4-R5 wait
    - R0 finishes → R4 admitted
    - R1 finishes → R5 admitted

Print queue sizes at each step:
    waiting=N, running=N, swapped=N

== PART 2: The Schedule Step ==

Detailed walkthrough of one schedule() call:
    Phase 1: check running (N requests, N tokens budget used)
    Phase 2: check swapped (empty)
    Phase 3: admit from waiting (N requests admitted, N tokens budget used)
    Result: SchedulerOutput { new=N, running=N, preempted=0 }

== PART 3: Memory-Aware Admission ==

10 blocks total, requests need 2 blocks each:
    - Admit 5 requests → 0 free blocks
    - Next request: can_allocate(2) = false → stays in waiting
    - Show the admission check in action

== PART 4: When Memory Runs Out — Preemption ==

Running requests grow and need new blocks:
    - No free blocks → scheduler must preempt
    - Preempt last-admitted (lowest priority in FCFS)
    - Show Swap policy: moved to swapped queue
    - Show Recompute policy: blocks freed, moved to waiting

== PART 5: A Day in the Scheduler's Life ==

5-step simulation:
    Step 1: Admit R0-R3
    Step 2: All decode, no issues
    Step 3: R0 finishes → R4 admitted
    Step 4: Memory pressure → preempt R4
    Step 5: R1 finishes → R4 re-admitted from swapped

Print full state after each step:
    waiting: [...]
    running: [...]
    swapped: [...]

== PART 6: The Scheduler Contract ==

Show SchedulerOutput for each step of the simulation:
    Step N: SchedulerOutput {
        new_requests: [...],
        running_requests: [...],
        preempted_ids: [...],
        num_prefill_tokens: N,
        num_decode_tokens: N,
    }

=== OUTPUT FORMAT ===

6 sections using the standard section() format (78 '=' chars):

PART 1: Three Queues and a GPU
PART 2: The Schedule Step
PART 3: Memory-Aware Admission
PART 4: When Memory Runs Out — Preemption
PART 5: A Day in the Scheduler's Life
PART 6: The Scheduler Contract

Closing: "Chapter 12 complete. Next: The Engine Loop (ch13)"

=== VALIDATION ===

Your output should contain:
- "PART 1" through "PART 6"
- "waiting", "running", "swapped" (three queues)
- "schedule" (the schedule() method)
- "preempt" or "preemption" (preemption policy)
- "block" or "memory" (memory-aware admission)
- "new_requests" or "prefill" and "running" or "decode" (SchedulerOutput)
- "preempted" (preempted IDs)
- Multi-step simulation showing queue state changes
- "Chapter 12 complete"

=== WHAT TO PRODUCE ===

1. src/types.[ext] — updated with Swapped state, SchedulerOutput
2. src/scheduler/mod.[ext] — Scheduler trait
3. src/scheduler/fcfs.[ext] — FcfsScheduler implementation
4. examples/ch12_the_scheduler.[ext] — the demo program

After this chapter:
  src/
    types.[ext]             (MODIFIED — Swapped state, SchedulerOutput)
    scheduler/
      mod.[ext]             (NEW — Scheduler trait)
      fcfs.[ext]            (NEW — FcfsScheduler)
  examples/
    ch12_the_scheduler.[ext]  (NEW)
```
