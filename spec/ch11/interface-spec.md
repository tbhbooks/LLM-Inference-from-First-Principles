# Chapter 11: Interface Specification

## Overview

This chapter introduces **continuous batching** (iteration-level scheduling), the technique from the Orca paper that replaces static batching in LLM inference. The code component is small: a `SequenceStatus` enum and a `SequenceGroup` type that will be used by the scheduler in Chapter 12.

The demo program simulates both static and continuous batching to show the throughput difference.

## Dependencies

- **Chapter 10**: PagedAttention (block-based memory). No direct code dependency for this chapter.
- **Chapters 3-8**: Existing module structure. `types` module gets new types.

## New Data Types

### SequenceStatus (enum)

Tracks the lifecycle of a request through the system.

| Variant | Meaning |
|---------|---------|
| `Waiting` | Queued but not yet scheduled for any GPU work |
| `Running` | Actively being processed (prefill or decode) |
| `Finished` | Generation complete (hit stop token or max length) |

**Invariants:**
- A sequence transitions `Waiting -> Running -> Finished`
- A sequence never moves backward (Finished -> Running is illegal)
- Only `Running` sequences consume KV cache blocks

**State machine:**

```
Waiting ──(scheduled)──> Running ──(done)──> Finished
```

Note: Chapter 12 will add a `Swapped` state for preemption. For now, three states are sufficient.

### SequenceGroup

Groups related sequences (e.g., a single user request that may produce multiple candidate outputs via beam search). For now, most groups contain exactly one sequence.

| Field | Type | Description |
|-------|------|-------------|
| `group_id` | GroupId (int) | Unique identifier |
| `sequences` | list of Sequence | The sequences in this group (usually 1) |
| `arrival_time` | float/timestamp | When the request arrived |
| `status` | SequenceStatus | Current status of the group |

A `Sequence` within the group has:

| Field | Type | Description |
|-------|------|-------------|
| `seq_id` | SeqId (int) | Unique sequence identifier |
| `prompt_tokens` | list of int | The input token IDs |
| `output_tokens` | list of int | Generated tokens so far |
| `status` | SequenceStatus | Status of this specific sequence |

**Invariants:**
- `group_id` is unique across all groups
- `seq_id` is unique across all sequences
- A group is `Finished` when all its sequences are `Finished`
- `prompt_tokens` is immutable after creation
- `output_tokens` grows by one each decode iteration

### Helper methods

#### `Sequence.num_tokens() -> int`

Total tokens (prompt + generated).

```
return len(prompt_tokens) + len(output_tokens)
```

#### `Sequence.is_prefill() -> bool`

True if no output tokens have been generated yet (still processing the prompt).

```
return len(output_tokens) == 0
```

#### `SequenceGroup.is_finished() -> bool`

```
return all(seq.status == Finished for seq in sequences)
```

## Simulation Types (demo only)

These types exist only in the demo program, not in the engine.

### SimRequest

A simulated request for the batching demo.

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Request identifier |
| `prompt_len` | int | Number of prompt tokens |
| `output_len` | int | Number of tokens to generate |
| `tokens_generated` | int | Tokens generated so far (starts at 0) |
| `status` | SequenceStatus | Current status |

### StaticBatchResult

| Field | Type | Description |
|-------|------|-------------|
| `total_iterations` | int | Total GPU iterations used |
| `total_tokens_generated` | int | Total output tokens across all requests |
| `gpu_idle_slots` | int | Batch slots wasted on finished requests |

### ContinuousBatchResult

| Field | Type | Description |
|-------|------|-------------|
| `total_iterations` | int | Total GPU iterations used |
| `total_tokens_generated` | int | Total output tokens across all requests |
| `requests_swapped_in` | int | How many requests entered mid-batch |

## Constants (for demo program)

```
MAX_BATCH_SIZE = 4          // GPU can process 4 sequences per iteration
NUM_REQUESTS = 8            // Total requests to serve
```

## Demo Scenarios

### Scenario 1: The Static Batching Problem

Show how static batching works: form a batch, run it until ALL requests finish, then form the next batch.

Requests:
```
Request 0: prompt=10, output=3    (finishes fast)
Request 1: prompt=10, output=8    (medium)
Request 2: prompt=10, output=2    (finishes fastest)
Request 3: prompt=10, output=10   (slowest in batch)
Request 4: prompt=10, output=4    (waiting)
Request 5: prompt=10, output=6    (waiting)
Request 6: prompt=10, output=5    (waiting)
Request 7: prompt=10, output=7    (waiting)
```

Static batching:
- Batch 1: requests [0,1,2,3]. Must run for 10 iterations (max output_len in batch).
  - Request 2 finishes at iteration 2, idles for 8 iterations
  - Request 0 finishes at iteration 3, idles for 7 iterations
  - Request 1 finishes at iteration 8, idles for 2 iterations
  - Request 3 finishes at iteration 10
  - Total idle slots: 8 + 7 + 2 = 17
- Batch 2: requests [4,5,6,7]. Must run for 7 iterations.
  - Similar idle pattern

Show: per-iteration timeline with which slots are active vs idle.

### Scenario 2: Continuous Batching

Same 8 requests, but now the scheduler checks every iteration:
- When a request finishes, its slot opens immediately
- A waiting request fills the slot next iteration
- The batch stays full (or as full as possible)

Show: per-iteration timeline. Requests enter and leave mid-batch.

### Scenario 3: Mixed Prefill + Decode

Show that continuous batching naturally handles mixed batches:
- Some slots run prefill (processing prompt tokens)
- Other slots run decode (generating one token)
- The scheduler manages both in the same iteration

### Scenario 4: Throughput Comparison

Side-by-side comparison:

| Metric | Static | Continuous |
|--------|--------|-----------|
| Total iterations | higher | lower |
| GPU idle slots | many | near zero |
| Avg time-to-first-token | higher (wait for batch) | lower (join immediately) |
| Throughput (tokens/iteration) | lower | higher |

### Scenario 5: SequenceStatus + SequenceGroup

Demonstrate the new types:
- Create SequenceGroups with different statuses
- Show transitions: Waiting -> Running -> Finished
- Show a group with is_prefill() and num_tokens()

### Scenario 6: Scaling the Advantage

Show how the throughput advantage of continuous batching grows with:
- More requests (larger queue)
- More variance in output lengths

## Output Sections

| Section | Title |
|---------|-------|
| PART 1 | The Waiting Game — Static Batching |
| PART 2 | What If Nobody Had to Wait? — Continuous Batching |
| PART 3 | The Iteration-Level View |
| PART 4 | Mixed Batches — Prefill Meets Decode |
| PART 5 | The New Types — SequenceStatus and SequenceGroup |
| PART 6 | The Throughput Multiplier |

## Validation Rules

1. All 6 section headers present ("PART 1" through "PART 6")
2. "static" batching mentioned (the baseline)
3. "continuous" batching mentioned (the improvement)
4. "iteration" mentioned (iteration-level scheduling)
5. "prefill" and "decode" mentioned (mixed batch handling)
6. SequenceStatus states shown: "Waiting", "Running", "Finished"
7. Throughput comparison showing continuous > static
8. "Chapter 11 complete" closing
