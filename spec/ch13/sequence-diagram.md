# Chapter 13: Sequence Diagram

## Single Request — Full Lifecycle

```mermaid
sequenceDiagram
    participant Client
    participant Engine
    participant Tok as Tokenizer
    participant Sched as Scheduler
    participant Alloc as BlockAllocator
    participant Model
    participant Samp as Sampler

    Client->>Engine: add_request("What is AI?", max_tokens=5)
    Engine->>Tok: encode("What is AI?")
    Tok-->>Engine: [2061, 318, 9552, 30]
    Engine->>Sched: add_sequence(A, blocks_needed=1)

    Note over Engine: step() — iteration 1 (prefill)

    Engine->>Sched: schedule()
    Sched-->>Engine: {scheduled: [A], newly_scheduled: [A]}
    Engine->>Alloc: allocate()
    Alloc-->>Engine: block_0
    Note over Engine: prepare_inputs:<br/>tokens=[2061,318,9552,30]<br/>positions=[0,1,2,3]
    Engine->>Model: forward(batch)
    Model-->>Engine: logits
    Engine->>Samp: sample(logits)
    Samp-->>Engine: token 42
    Note over Engine: A: 5 tokens (4+1)

    Note over Engine: step() — iterations 2-4 (decode)

    loop 3 more decode steps
        Engine->>Sched: schedule()
        Sched-->>Engine: {scheduled: [A]}
        Engine->>Model: forward(1 token)
        Model-->>Engine: logits
        Engine->>Samp: sample(logits)
        Samp-->>Engine: new token
    end

    Note over Engine: step() — iteration 5 (final)

    Engine->>Sched: schedule()
    Engine->>Model: forward(1 token)
    Engine->>Samp: sample(logits)
    Samp-->>Engine: token 29
    Note over Engine: A: 9 tokens (4+5) → FINISHED
    Engine->>Alloc: free(block_0)
    Engine->>Sched: finish_sequence(A)
    Engine->>Tok: decode([42, 87, 15, 63, 29])
    Tok-->>Engine: "generated text"
    Engine-->>Client: CompletedRequest(A)
```
**Figure 13.4** — Full lifecycle of a single request through the engine. Five step() iterations: one prefill, four decodes. Blocks are allocated on admission and freed on completion.

## Multiple Requests — Staggered Batching

```mermaid
sequenceDiagram
    participant Engine
    participant Sched as Scheduler
    participant Model
    participant Alloc as BlockAllocator

    Note over Engine: Before step 1: add A

    Engine->>Sched: schedule()
    Sched-->>Engine: [A(prefill)]
    Engine->>Alloc: allocate 1 block for A
    Engine->>Model: forward([A])
    Note over Engine: A gets first token

    Note over Engine: Before step 2: add B

    Engine->>Sched: schedule()
    Sched-->>Engine: [A(decode), B(prefill)]
    Engine->>Alloc: allocate 1 block for B
    Engine->>Model: forward([A, B])
    Note over Engine: A+B each get a token

    Note over Engine: Before step 3: add C

    Engine->>Sched: schedule()
    Sched-->>Engine: [A(decode), B(decode), C(prefill)]
    Engine->>Alloc: allocate 1 block for C
    Engine->>Model: forward([A, B, C])
    Note over Engine: Batch of 3

    Note over Engine: Step 4: B finishes

    Engine->>Sched: schedule()
    Sched-->>Engine: [A, B, C] (all decode)
    Engine->>Model: forward([A, B, C])
    Note over Engine: B → FINISHED
    Engine->>Alloc: free B's block
    Engine->>Sched: finish_sequence(B)

    Note over Engine: Step 5: A finishes

    Engine->>Sched: schedule()
    Sched-->>Engine: [A, C]
    Engine->>Model: forward([A, C])
    Note over Engine: A → FINISHED
    Engine->>Alloc: free A's block
```
**Figure 13.5** — Staggered arrivals. Requests join the batch at different steps. The batch grows as requests arrive and shrinks as requests finish. The scheduler and block allocator handle the bookkeeping; the engine just calls step().
