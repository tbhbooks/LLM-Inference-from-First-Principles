# Chapter 2: Request Lifecycle Sequence Diagram

## The 9-Step Request Lifecycle

A single request -- "Explain PagedAttention" -- traced through rvllm's modules.

```mermaid
sequenceDiagram
    actor Client
    participant API as api/
    participant TOK as tokenizer/
    participant SCHED as scheduler/
    participant MEM as memory/
    participant MODEL as model/
    participant SAMP as sampling/
    participant ENGINE as engine/

    Note over Client,ENGINE: Step 1: ARRIVE
    Client->>API: POST /v1/completions<br/>{"prompt": "Explain PagedAttention", ...}
    API->>API: Validate request, extract parameters

    Note over Client,ENGINE: Step 2: TOKENIZE
    API->>TOK: "Explain PagedAttention"
    TOK-->>API: [50872, 7873, 58662]

    Note over Client,ENGINE: Step 3: ENQUEUE
    API->>SCHED: Add Seq(id=42, tokens=[50872, 7873, 58662])
    SCHED->>SCHED: Push to WAITING queue

    rect rgb(230, 245, 255)
        Note over Client,ENGINE: Steps 4-8 LOOP (repeat for each new token)

        Note over SCHED,MEM: Step 4: SCHEDULE
        SCHED->>SCHED: Can Seq(42) fit in running batch?<br/>Enough free KV blocks?<br/>Preempt lower-priority?
        SCHED->>SCHED: Move Seq(42): WAITING -> RUNNING

        Note over MEM: Step 5: ALLOCATE
        SCHED->>MEM: Allocate blocks for Seq(42)
        MEM->>MEM: Assign block_7 (3 tokens fit in block_size=16)<br/>Free blocks: 254 -> 253
        MEM-->>SCHED: Block table: {42: [block_7]}

        Note over MODEL: Step 6: EXECUTE
        ENGINE->>MODEL: Forward pass with batch
        MODEL->>MODEL: Embed tokens -> transformer layers -> logits

        Note over SAMP: Step 7: SAMPLE
        MODEL->>SAMP: logits [0.01, 0.03, ..., 0.42, ...]
        SAMP->>SAMP: temperature=0 (greedy): argmax -> token 1334
        SAMP-->>ENGINE: token 1334 ("Paged")

        Note over ENGINE: Step 8: UPDATE
        ENGINE->>ENGINE: Append token: Seq(42).tokens = [..., 1334]<br/>Stop? No (not EOS, not max_tokens)
    end

    Note over Client,ENGINE: Step 9: RESPOND
    ENGINE->>API: New token(s) ready
    API-->>Client: data: {"choices": [{"text": "Paged"}]}

    Note over Client,ENGINE: Loop terminates when:<br/>EOS token | max_tokens reached | client disconnects
```

## Step Summary

| Step | Name | Module | Action |
|------|------|--------|--------|
| 1 | ARRIVE | api/ | HTTP POST received, request validated |
| 2 | TOKENIZE | tokenizer/ | Text converted to token IDs |
| 3 | ENQUEUE | scheduler/ | Sequence added to WAITING queue |
| 4 | SCHEDULE | scheduler/ | Sequence moved WAITING -> RUNNING |
| 5 | ALLOCATE | memory/ | KV cache blocks assigned |
| 6 | EXECUTE | model/ | Forward pass produces logits |
| 7 | SAMPLE | sampling/ | Next token selected from logits |
| 8 | UPDATE | engine/ | Token appended, stop criteria checked |
| 9 | RESPOND | api/ | Token streamed back to client |

Steps 4-8 repeat for every generated token until EOS, max_tokens, or client disconnect.
