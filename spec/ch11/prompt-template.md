# Chapter 11 -- LLM Prompt Template

Copy and paste this prompt into your LLM of choice to generate a working
implementation. This chapter introduces continuous batching (iteration-level
scheduling) and adds SequenceStatus / SequenceGroup types.

---

## Prompt

```
I am building an LLM inference engine called "rvllm" as a learning project.
This is Chapter 11. I have a working MVP from Chapters 1-8 (model loading,
forward pass, KV cache, greedy generation), a memory fragmentation simulator
from Chapter 9, and PagedAttention from Chapter 10.

Now I need to understand CONTINUOUS BATCHING — the iteration-level scheduling
technique from the Orca paper that replaces static batching.

This chapter builds a standalone demo that simulates both static and continuous
batching to show the throughput difference. It also introduces SequenceStatus
and SequenceGroup types that the scheduler will use in Chapter 12.

TARGET LANGUAGE: [Rust / Python / Go / your choice]

=== WHAT TO CREATE ===

NEW FILES:
  examples/ch11_continuous_batching.[ext]    <-- the program for this chapter

MODIFY:
  src/types.[ext]                            <-- add SequenceStatus, SequenceGroup

KEEP UNCHANGED:
  Everything from chapters 1-10.

=== NEW TYPES ===

== SequenceStatus (enum) ==

Tracks the lifecycle of a request:

    Waiting    — queued, not yet scheduled
    Running    — actively being processed (prefill or decode)
    Finished   — generation complete

Transitions: Waiting -> Running -> Finished (never backward)

== SequenceGroup ==

Groups related sequences for a single user request.

Fields:
    group_id: int             — unique identifier
    sequences: list           — the sequences in this group (usually 1)
    arrival_time: timestamp   — when the request arrived
    status: SequenceStatus    — current status

== Sequence (within a group) ==

Fields:
    seq_id: int               — unique sequence identifier
    prompt_tokens: list[int]  — input token IDs (immutable after creation)
    output_tokens: list[int]  — generated tokens so far
    status: SequenceStatus    — current status

Methods:
    num_tokens() -> int       — len(prompt_tokens) + len(output_tokens)
    is_prefill() -> bool      — len(output_tokens) == 0

=== SIMULATION TYPES (demo only) ===

== SimRequest ==

A simulated request for the batching demo.

Fields:
    id: int                   — request identifier
    prompt_len: int           — prompt token count
    output_len: int           — tokens to generate
    tokens_generated: int     — generated so far (starts 0)
    status: SequenceStatus    — current status

=== THE DEMO PROGRAM ===

The demo simulates batching strategies to compare throughput.

Constants:
    MAX_BATCH_SIZE = 4
    NUM_REQUESTS = 8

Requests (all have prompt_len=10, vary in output_len):
    Request 0: output=3
    Request 1: output=8
    Request 2: output=2
    Request 3: output=10
    Request 4: output=4
    Request 5: output=6
    Request 6: output=5
    Request 7: output=7

== PART 1: The Waiting Game — Static Batching ==

Static batching: form a batch of up to MAX_BATCH_SIZE requests. Run the
entire batch until ALL requests in it finish (i.e., until the slowest one
completes). Then form the next batch.

For each iteration, print which requests are active and which are idle
(finished but still occupying a batch slot).

Batch 1: [0,1,2,3], runs for max(3,8,2,10) = 10 iterations
  - Request 2 finishes at iteration 2, idles for 8 more
  - Request 0 finishes at iteration 3, idles for 7 more
  - Request 1 finishes at iteration 8, idles for 2 more
  - Request 3 finishes at iteration 10

Batch 2: [4,5,6,7], runs for max(4,6,5,7) = 7 iterations
  - Similar tracking

Count total iterations, total idle slots, effective utilization.

== PART 2: What If Nobody Had to Wait? — Continuous Batching ==

Same 8 requests, but now check every iteration:
  - If a request finishes, remove it from the batch
  - If a waiting request exists, add it to the batch
  - The batch stays full (or as full as the queue allows)

Show per-iteration timeline: which requests are in each slot.
A request might enter at iteration 3 when another finishes.

Count the same metrics for comparison.

== PART 3: The Iteration-Level View ==

For 5-6 iterations, show the detailed state:
  - Which requests are in the batch
  - Which are generating (decode)
  - Which just entered (prefill)
  - Which just finished and left

Make it clear that decisions happen EVERY iteration, not per-batch.

== PART 4: Mixed Batches — Prefill Meets Decode ==

When a new request joins an ongoing batch:
  - The new request needs prefill (process all prompt tokens)
  - Existing requests are in decode (generate one token)
  - They share the same GPU iteration

Show a mixed batch: e.g., 3 requests decoding + 1 request in prefill.
Note: prefill is more compute-intensive than decode, so the scheduler
must account for this. (Chapter 12 handles the details.)

== PART 5: The New Types — SequenceStatus and SequenceGroup ==

Demonstrate the engine types:
  - Create sequences with different statuses
  - Transition: Waiting -> Running -> Finished
  - Show num_tokens(), is_prefill()
  - Show a SequenceGroup containing one sequence

Print the state at each transition.

== PART 6: The Throughput Multiplier ==

Side-by-side comparison table:

    Metric                      Static      Continuous
    Total iterations            ??          ??
    GPU idle slots              ??          near 0
    Effective utilization       ??%         ??%
    Throughput (tok/iter)       ??          ??

The continuous batching numbers should be significantly better.

End with: why this matters at scale. With 100 requests and high variance
in output lengths, continuous batching can 2-3x throughput.

=== OUTPUT FORMAT ===

6 sections using the standard section() format (78 '=' chars):

PART 1: The Waiting Game — Static Batching
PART 2: What If Nobody Had to Wait? — Continuous Batching
PART 3: The Iteration-Level View
PART 4: Mixed Batches — Prefill Meets Decode
PART 5: The New Types — SequenceStatus and SequenceGroup
PART 6: The Throughput Multiplier

Closing: "Chapter 11 complete. Next: The Scheduler (ch12)"

=== VALIDATION ===

Your output should contain:
- "PART 1" through "PART 6"
- "static" (static batching)
- "continuous" (continuous batching)
- "iteration" (iteration-level scheduling)
- "prefill" and "decode" (mixed batch handling)
- "Waiting", "Running", "Finished" (SequenceStatus states)
- Throughput comparison showing continuous batching wins
- "Chapter 11 complete"

=== WHAT TO PRODUCE ===

1. src/types.[ext] — updated with SequenceStatus, SequenceGroup, Sequence
2. examples/ch11_continuous_batching.[ext] — the demo program

After this chapter:
  src/
    types.[ext]        (MODIFIED — new types added)
  examples/
    ch11_continuous_batching.[ext]  (NEW)
```
