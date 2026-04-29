# Chapter 13: Interface Specification

## Overview

This chapter builds the **Engine** — the orchestrator that ties scheduler, model, tokenizer, sampler, and block allocator into a single step-by-step execution loop. The engine does not implement any of these components; it *conducts* them. One method — `step()` — drives the entire inference pipeline: schedule, prepare inputs, forward pass, sample, update sequences, check finish conditions.

This is the chapter where every component from the book meets every other component for the first time.

## Dependencies

- **Chapter 4-6**: Model (forward pass, KV cache). The engine calls `model.forward()`.
- **Chapter 7**: Sampler (greedy). The engine calls `sampler.sample()`.
- **Chapter 4**: Tokenizer (encode/decode). The engine calls `tokenizer.decode()`.
- **Chapter 10**: BlockAllocator and BlockTable. The engine allocates blocks for new sequences.
- **Chapter 12**: Scheduler (FCFS, three queues, preemption). The engine calls `scheduler.schedule()`.

## New Data Types

### EngineConfig

Configuration for the engine.

| Field | Type | Description |
|-------|------|-------------|
| `max_num_seqs` | int | Maximum concurrent sequences (passed to scheduler) |
| `max_tokens_per_step` | int | Token budget per step (passed to scheduler) |
| `block_size` | int | Tokens per block (passed to block allocator) |
| `num_gpu_blocks` | int | Total GPU blocks available |
| `max_model_len` | int | Maximum sequence length the model supports |

### SequenceData

Per-sequence tracking data held by the engine.

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | RequestId | Unique identifier for this request |
| `token_ids` | list of int | All token IDs (prompt + generated) |
| `prompt_len` | int | Length of the original prompt |
| `status` | SequenceStatus | Current lifecycle state |
| `block_table` | BlockTable | This sequence's block-to-physical mapping (ch10) |
| `max_tokens` | int | Maximum tokens to generate |

### SequenceStatus (enum)

```
Waiting    — queued, not yet scheduled
Running    — actively generating tokens
Finished   — hit stop condition or max length
```

### StepOutput

What `step()` returns after one iteration.

| Field | Type | Description |
|-------|------|-------------|
| `completed` | list of CompletedRequest | Requests that finished this step |
| `num_running` | int | Sequences still generating |
| `num_waiting` | int | Sequences still queued |
| `num_scheduled` | int | How many sequences ran this step |
| `num_tokens_generated` | int | Total new tokens produced this step |

### CompletedRequest

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | RequestId | Which request |
| `output_text` | string | Decoded generated text |
| `prompt_text` | string | Original prompt |
| `num_tokens` | int | Total tokens generated |

## Engine (struct)

The central orchestrator.

| Field | Type | Description |
|-------|------|-------------|
| `config` | EngineConfig | Engine configuration |
| `scheduler` | Scheduler | From ch12 — manages waiting/running/swapped queues |
| `model` | Model (trait) | From ch05-06 — runs forward pass |
| `tokenizer` | TokenizerBackend (trait) | From ch04 — encodes/decodes text |
| `sampler` | Sampler (trait) | From ch07 — selects next token |
| `block_allocator` | BlockAllocator (trait) | From ch10 — manages KV cache blocks |
| `sequences` | map of RequestId -> SequenceData | All tracked sequences |

### `add_request(request_id, prompt_text, max_tokens) -> Result`

Add a new request to the engine.

```
precondition: request_id is unique (not already tracked)

1. token_ids = tokenizer.encode(prompt_text)
2. Create SequenceData with status = Waiting, prompt_len = len(token_ids)
3. Compute blocks_needed = ceil(len(token_ids) / block_size)
4. scheduler.add_sequence(request_id, blocks_needed)
5. Store sequence in sequences map
```

### `step() -> StepOutput`

The core loop iteration. This is the heart of the engine.

```
// Phase 1: Ask the scheduler what to run
scheduler_output = scheduler.schedule()

// Phase 2: Allocate blocks for newly admitted sequences
for seq_id in scheduler_output.newly_scheduled:
    blocks_needed = ceil(len(sequences[seq_id].token_ids) / block_size)
    for i in 0..blocks_needed:
        block_id = block_allocator.allocate()
        sequences[seq_id].block_table.append_block(block_id)
    sequences[seq_id].status = Running

// Phase 3: Prepare model inputs from scheduled sequences
(token_ids, positions, block_tables) = prepare_inputs(scheduler_output)

// Phase 4: Run the forward pass
logits = model.forward(token_ids, positions, block_tables)

// Phase 5: Sample next tokens
new_token_ids = sampler.sample(logits)    // one token per sequence in batch

// Phase 6: Update sequences and check finish conditions
completed = []
for (seq_id, new_token) in zip(scheduler_output.scheduled_seq_ids, new_token_ids):
    seq = sequences[seq_id]
    seq.token_ids.append(new_token)

    // Allocate a new block if the current last block is full
    if needs_new_block(seq):
        block_id = block_allocator.allocate()
        seq.block_table.append_block(block_id)

    // Check finish conditions
    if is_finished(seq, new_token):
        seq.status = Finished
        scheduler.finish_sequence(seq_id)
        free_blocks(seq)
        output_text = tokenizer.decode(seq.token_ids[seq.prompt_len:])
        completed.append(CompletedRequest(seq_id, output_text, ...))

return StepOutput(
    completed = completed,
    num_running = scheduler.num_running(),
    num_waiting = scheduler.num_waiting(),
    num_scheduled = len(scheduler_output.scheduled_seq_ids),
    num_tokens_generated = len(new_token_ids),
)
```

### `prepare_inputs(scheduler_output) -> (token_ids, positions, block_tables)`

Convert scheduler output into tensors the model can consume.

```
token_ids = []        // one token ID per scheduled sequence (decode) or prompt (prefill)
positions = []        // position index for each token
block_tables = []     // block table for each scheduled sequence

for seq_id in scheduler_output.scheduled_seq_ids:
    seq = sequences[seq_id]
    if seq is in prefill:
        // First time: feed all prompt tokens
        token_ids.append(seq.token_ids)
        positions.append([0, 1, 2, ..., len(seq.token_ids)-1])
    else:
        // Decode: feed only the last token
        token_ids.append([seq.token_ids[-1]])
        positions.append([len(seq.token_ids) - 1])
    block_tables.append(seq.block_table.block_ids)

return (token_ids, positions, block_tables)
```

### `is_finished(seq, new_token) -> bool`

Check if a sequence should stop generating.

```
// Stop if we hit the EOS token
if new_token == tokenizer.eos_token_id():
    return true

// Stop if we've generated max_tokens
generated = len(seq.token_ids) - seq.prompt_len
if generated >= seq.max_tokens:
    return true

// Stop if we've hit the model's max sequence length
if len(seq.token_ids) >= config.max_model_len:
    return true

return false
```

### `needs_new_block(seq) -> bool`

Check if appending a token requires allocating a new block.

```
total_tokens = len(seq.token_ids)
current_capacity = seq.block_table.num_tokens_capacity()
return total_tokens > current_capacity
```

### `free_blocks(seq)`

Return all blocks held by a finished sequence.

```
for block_id in seq.block_table.block_ids:
    block_allocator.free(block_id)
seq.block_table.block_ids = []
```

### `run(max_steps) -> list of CompletedRequest`

Convenience method: run the engine for up to `max_steps` iterations or until all requests are finished.

```
all_completed = []
for step_num in 0..max_steps:
    output = step()
    all_completed.extend(output.completed)
    if output.num_running == 0 and output.num_waiting == 0:
        break
return all_completed
```

## Helper Functions

### `blocks_needed(num_tokens, block_size) -> int`

```
return ceil(num_tokens / block_size)
```

## Constants (for demo program)

```
BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 20          // 320 token slots
MAX_NUM_SEQS = 4             // max 4 concurrent sequences
MAX_TOKENS_PER_STEP = 64     // token budget per step
MAX_MODEL_LEN = 128          // max sequence length
```

## Demo Scenarios

### Scenario 1: Single Request Lifecycle

One request flows through the complete engine:
- Add request "What is AI?" (4 tokens)
- Step 1: schedule → prefill → sample → update (first generated token)
- Steps 2-N: schedule → decode → sample → update
- Final step: finish condition met, request completes

Show each step with: what was scheduled, what token was produced, sequence length.

### Scenario 2: Multiple Requests, Staggered Arrivals

Three requests arrive at different times:
- Step 0: Add request A
- Step 1: Engine runs A (prefill)
- Step 2: Add request B, engine runs A+B
- Step 3: Add request C, engine runs A+B+C
- Steps 4+: All three running until they finish at different times

Show the batch composition at each step and when each request finishes.

### Scenario 3: Request Lifecycle Tracking

Show the status transitions for each request:
```
Request A: Waiting → Running → Finished (step 6)
Request B: Waiting → Running → Finished (step 8)
Request C: Waiting → Running → Finished (step 5)
```

### Scenario 4: Block Allocation During Generation

Track block allocation across steps:
- Show free blocks at each step
- Show when new blocks are allocated as sequences grow
- Show blocks freed when sequences complete

### Scenario 5: The Complete Pipeline Summary

A summary diagram showing all components and data flow:
- Request arrives → tokenize → scheduler → prepare inputs → forward → sample → update → repeat or finish

## Output Sections

| Section | Title |
|---------|-------|
| PART 1 | The Conductor — One Engine to Run Them All |
| PART 2 | Anatomy of a Step — Schedule, Forward, Sample, Update |
| PART 3 | Input Preparation — Wiring the Pieces |
| PART 4 | The Update Phase — Tokens, Blocks, and Finish |
| PART 5 | A Request's Full Journey |
| PART 6 | The Complete Pipeline |

## Validation Rules

1. All 6 section headers present: "PART 1" through "PART 6"
2. "step" mentioned (the core method)
3. "schedule" mentioned (scheduler invocation)
4. "forward" mentioned (model forward pass)
5. "sample" mentioned (sampling phase)
6. Request lifecycle shown (Waiting → Running → Finished)
7. Block allocation tracked (free blocks count changes)
8. Multiple requests shown running concurrently
9. "Chapter 13 complete" closing
