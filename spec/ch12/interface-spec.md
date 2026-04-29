# Chapter 12: Interface Specification

## Overview

This chapter implements the **scheduler** --- the component that decides which requests get GPU time at each iteration. It implements FCFS (first-come, first-served) scheduling with three queues (waiting, running, swapped), memory-aware admission via the BlockAllocator, and preemption when memory runs out.

The scheduler is the bridge between continuous batching (Chapter 11's concept) and the engine loop (Chapter 13). It produces a `SchedulerOutput` each iteration that tells the engine exactly what to run.

## Dependencies

- **Chapter 10**: BlockAllocator --- the scheduler calls `can_allocate()` and `num_free_blocks()` to check memory before admitting requests.
- **Chapter 11**: SequenceStatus, SequenceGroup --- the scheduler manages sequences through their lifecycle. This chapter adds a `Swapped` state to SequenceStatus.

## Updated Data Types

### SequenceStatus (updated enum)

Chapter 11 defined three states. This chapter adds `Swapped`:

| Variant | Meaning |
|---------|---------|
| `Waiting` | Queued but not yet scheduled for any GPU work |
| `Running` | Actively being processed (prefill or decode) |
| `Swapped` | Was running, preempted due to memory pressure, can resume later |
| `Finished` | Generation complete (hit stop token or max length) |

**State machine:**

```
Waiting ──(admitted)──> Running ──(done)──> Finished
                           │        ↑
                     (preempt)   (resume)
                           ↓        │
                         Swapped ───┘
```

**Invariants:**
- `Waiting -> Running` when scheduler admits a request (has memory)
- `Running -> Swapped` when scheduler preempts under memory pressure
- `Swapped -> Running` when memory becomes available
- `Running -> Finished` when stop token or max length reached
- `Finished` is terminal --- no transitions out

## New Data Types

### SchedulerConfig

Configuration for the scheduler.

| Field | Type | Description |
|-------|------|-------------|
| `max_num_seqs` | int | Maximum number of sequences in the running batch |
| `max_num_batched_tokens` | int | Maximum total tokens processed per iteration |
| `block_size` | int | Tokens per KV cache block (must match allocator) |
| `preemption_policy` | PreemptionPolicy | How to handle memory pressure |

### PreemptionPolicy (enum)

| Variant | Meaning |
|---------|---------|
| `Swap` | Move preempted request to swapped queue, keep its KV cache recoverable |
| `Recompute` | Discard KV cache, move back to waiting queue to redo prefill |

### SchedulerOutput

The result of one `schedule()` call. Tells the engine what to execute this iteration.

| Field | Type | Description |
|-------|------|-------------|
| `new_requests` | list of SequenceGroup | Sequences moving Waiting -> Running (need prefill) |
| `running_requests` | list of SequenceGroup | Sequences continuing to decode |
| `preempted_ids` | list of GroupId | Requests preempted this step (Running -> Swapped or Waiting) |
| `num_prefill_tokens` | int | Total prompt tokens to process this step |
| `num_decode_tokens` | int | Total decode tokens (1 per running decode sequence) |

**Invariants:**
- `len(new_requests) + len(running_requests) <= max_num_seqs`
- `num_prefill_tokens + num_decode_tokens <= max_num_batched_tokens`
- Every ID in `preempted_ids` was previously Running
- `num_prefill_tokens = sum(seq.num_tokens() for seq in new_requests if seq.is_prefill())`
- `num_decode_tokens = len(running_requests) + len(non-prefill new_requests)`

### Scheduler (trait/interface)

The main scheduling interface.

#### `schedule() -> SchedulerOutput`

The core method. Called once per engine iteration. Decides which requests run.

```
Algorithm:
    1. Resume swapped requests (if memory available)
    2. Continue running requests (check they still have memory for next token)
    3. Admit new waiting requests (FCFS, if memory and batch budget allow)
    4. If memory insufficient for running requests → preempt lowest priority
    5. Build and return SchedulerOutput
```

#### `add_request(group: SequenceGroup)`

Add a new request to the waiting queue.

```
precondition: group.status == Waiting
postcondition: group is appended to waiting queue (FCFS order)
```

#### `notify_finished(group_id: GroupId)`

Mark a request as finished. Free its resources.

```
precondition: group exists and status == Running
postcondition: group removed from running set, blocks freed
```

#### `num_waiting() -> int`

Number of requests in the waiting queue.

#### `num_running() -> int`

Number of requests currently running.

#### `num_swapped() -> int`

Number of requests in the swapped queue.

### FcfsScheduler (implements Scheduler)

First-come, first-served scheduler with three queues.

**State:**

| Field | Type | Description |
|-------|------|-------------|
| `waiting` | queue of SequenceGroup | FIFO queue, ordered by arrival_time |
| `running` | list of SequenceGroup | Currently active sequences |
| `swapped` | queue of SequenceGroup | Preempted sequences, FIFO order |
| `config` | SchedulerConfig | Scheduling parameters |
| `block_allocator` | BlockAllocator | Memory manager (from ch10) |

**Constructor: `FcfsScheduler(config: SchedulerConfig, block_allocator: BlockAllocator)`**

```
waiting = empty FIFO queue
running = empty list
swapped = empty FIFO queue
```

#### `schedule()` algorithm (detailed)

```
schedule():
    output = empty SchedulerOutput
    token_budget = config.max_num_batched_tokens
    seq_budget = config.max_num_seqs

    // Phase 1: Continue running requests
    // Each running request needs 1 new token slot (may need a new block)
    still_running = []
    for group in running:
        blocks_needed = blocks_for_next_token(group)    // 0 or 1
        if blocks_needed > 0 and not allocator.can_allocate(blocks_needed):
            // Memory pressure! Must preempt.
            preempt(group, output)
        else:
            if blocks_needed > 0:
                allocate_block_for(group)
            still_running.append(group)
            token_budget -= 1     // 1 decode token per running seq
            seq_budget -= 1

    running = still_running
    output.running_requests = still_running
    output.num_decode_tokens = len(still_running)

    // Phase 2: Try to resume swapped requests
    while swapped is not empty and seq_budget > 0:
        group = swapped.front()
        blocks_needed = blocks_to_resume(group)
        if not allocator.can_allocate(blocks_needed):
            break    // can't resume any swapped — stop trying
        if token_budget < 1:
            break
        swapped.pop_front()
        restore_blocks(group)
        group.status = Running
        running.append(group)
        output.running_requests.append(group)
        output.num_decode_tokens += 1
        token_budget -= 1
        seq_budget -= 1

    // Phase 3: Admit new waiting requests (FCFS)
    while waiting is not empty and seq_budget > 0:
        group = waiting.front()
        num_prompt_tokens = group.sequences[0].num_tokens()
        blocks_needed = ceil(num_prompt_tokens / block_size)
        if not allocator.can_allocate(blocks_needed):
            break    // no memory for this request — stop admitting
        if token_budget < num_prompt_tokens:
            break    // exceeds token budget this iteration
        waiting.pop_front()
        allocate_initial_blocks(group, blocks_needed)
        group.status = Running
        running.append(group)
        output.new_requests.append(group)
        output.num_prefill_tokens += num_prompt_tokens
        token_budget -= num_prompt_tokens
        seq_budget -= 1

    return output
```

#### `preempt(group, output)` helper

```
preempt(group, output):
    if config.preemption_policy == Swap:
        group.status = Swapped
        swapped.push_back(group)
        // keep block table intact for later resume
    else:  // Recompute
        group.status = Waiting
        free_all_blocks(group)
        waiting.push_front(group)    // put at front so it gets readmitted first
    output.preempted_ids.append(group.group_id)
```

#### `blocks_for_next_token(group) -> int`

Check if the current last block has room for one more token.

```
seq = group.sequences[0]
current_tokens = seq.num_tokens()
current_blocks = ceil(current_tokens / block_size)
next_blocks = ceil((current_tokens + 1) / block_size)
return next_blocks - current_blocks    // 0 or 1
```

## Constants (for demo program)

```
NUM_BLOCKS = 10              // 10 physical blocks in the pool
BLOCK_SIZE = 16              // 16 tokens per block
MAX_NUM_SEQS = 4             // max 4 sequences running at once
MAX_NUM_BATCHED_TOKENS = 64  // max 64 tokens processed per iteration
```

## Demo Scenarios

### Scenario 1: Three Queues and a GPU

Show the three-queue architecture: waiting, running, swapped. Start with 6 requests, max_num_seqs=4. Show requests moving from waiting to running as slots open.

Requests:
```
Request 0: prompt=4 tokens    (1 block)
Request 1: prompt=4 tokens    (1 block)
Request 2: prompt=4 tokens    (1 block)
Request 3: prompt=4 tokens    (1 block)
Request 4: prompt=4 tokens    (1 block)   — waiting
Request 5: prompt=4 tokens    (1 block)   — waiting
```

Step 1: schedule() admits R0-R3 (batch full at max_num_seqs=4). R4, R5 remain waiting.
Step 2: R0 finishes. schedule() admits R4. R5 still waiting.
Step 3: R1 finishes. schedule() admits R5.

### Scenario 2: The Schedule Step

Detailed walkthrough of one schedule() call showing all three phases:
1. Check running requests can continue
2. Try to resume swapped (none swapped in this scenario)
3. Admit new requests from waiting queue

Show token and sequence budgets being consumed.

### Scenario 3: Memory-Aware Admission

Show what happens when memory limits admission:
- 10 blocks total, each request needs 2 blocks
- After admitting 4 requests (8 blocks), only 2 blocks remain
- Next request needs 2 blocks — admitted
- Next request needs 2 blocks — blocked (0 free blocks)
- Request waits until a running request finishes and frees blocks

### Scenario 4: Preemption

Show memory pressure triggering preemption:
- Running requests have grown their KV caches and need new blocks
- No free blocks available
- Scheduler preempts the lowest-priority (last-admitted) running request
- Preempted request moves to swapped queue
- Freed blocks allow other running requests to continue

Show both Swap and Recompute policies.

### Scenario 5: Multi-Step Simulation

5-step simulation showing the full lifecycle:
```
Step 1: Admit R0-R3, R4-R5 waiting
Step 2: All running decode, no preemption needed
Step 3: R0 finishes, R4 admitted
Step 4: Memory pressure — preempt R4, R1 continues
Step 5: R1 finishes, R4 re-admitted from swapped
```

Show queue states at each step.

### Scenario 6: The Scheduler Contract (SchedulerOutput)

Show the SchedulerOutput structure for each step of the simulation:
- new_requests list
- running_requests list
- preempted_ids list
- token counts

## Output Sections

| Section | Title |
|---------|-------|
| PART 1 | Three Queues and a GPU |
| PART 2 | The Schedule Step |
| PART 3 | Memory-Aware Admission |
| PART 4 | When Memory Runs Out — Preemption |
| PART 5 | A Day in the Scheduler's Life |
| PART 6 | The Scheduler Contract |

## Validation Rules

1. All 6 section headers present ("PART 1" through "PART 6")
2. Three queues mentioned: "waiting", "running", "swapped"
3. "schedule" mentioned (the schedule() method)
4. "preempt" or "preemption" mentioned
5. "block" or "memory" mentioned (memory-aware admission)
6. SchedulerOutput fields shown: "new_requests" or "prefill", "running" or "decode", "preempted"
7. Multi-step simulation with queue state changes
8. "Chapter 12 complete" closing
