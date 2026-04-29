# Chapter 11: Sequence Diagram

## Static Batching Flow

```mermaid
sequenceDiagram
    participant Q as Request Queue
    participant Sched as Scheduler
    participant GPU as GPU Batch<br/>(4 slots)

    Note over Q: 8 requests waiting

    Q->>Sched: take 4 requests [R0, R1, R2, R3]
    Sched->>GPU: form batch (4 slots full)

    loop 10 iterations (max output_len)
        GPU->>GPU: generate 1 token per active request
        Note over GPU: R2 finishes at iter 2<br/>R0 finishes at iter 3<br/>R1 finishes at iter 8<br/>Slots idle until iter 10
    end

    GPU->>Sched: batch complete
    Note over GPU: All 4 slots free

    Q->>Sched: take 4 requests [R4, R5, R6, R7]
    Sched->>GPU: form batch (4 slots full)

    loop 7 iterations
        GPU->>GPU: generate 1 token per active request
    end

    GPU->>Sched: batch complete
    Note over Sched: Total: 17 iterations<br/>23 idle slot-iterations
```
**Figure 11.5** --- Static batching sequence. The scheduler waits for the entire batch to finish before forming a new one. Fast requests idle while waiting for the slowest.

## Continuous Batching Flow

```mermaid
sequenceDiagram
    participant Q as Request Queue
    participant Sched as Scheduler
    participant GPU as GPU Batch<br/>(4 slots)

    Note over Q: 8 requests waiting

    Q->>Sched: initial fill [R0, R1, R2, R3]
    Sched->>GPU: batch = [R0, R1, R2, R3]

    GPU->>GPU: iteration 1

    GPU->>GPU: iteration 2 → R2 finishes
    GPU->>Sched: R2 done
    Q->>Sched: R4 available
    Sched->>GPU: replace R2 with R4

    GPU->>GPU: iteration 3 → R0 finishes
    GPU->>Sched: R0 done
    Q->>Sched: R5 available
    Sched->>GPU: replace R0 with R5

    Note over GPU: batch = [R5, R1, R4, R3]

    GPU->>GPU: iterations 4-6 → R4 finishes at 6
    GPU->>Sched: R4 done
    Q->>Sched: R6 available
    Sched->>GPU: replace R4 with R6

    GPU->>GPU: iterations 7-8 → R1 finishes at 8
    GPU->>Sched: R1 done
    Q->>Sched: R7 available
    Sched->>GPU: replace R1 with R7

    GPU->>GPU: iterations 9-15 → remaining finish
    Note over Sched: Total: 15 iterations<br/>15 idle slot-iterations
```
**Figure 11.6** --- Continuous batching sequence. The scheduler checks every iteration. When a request finishes, it is immediately replaced by a waiting request. The batch stays full as long as requests are queued.

## Sequence Lifecycle

```mermaid
sequenceDiagram
    participant User as User Request
    participant SG as SequenceGroup
    participant Seq as Sequence
    participant Sched as Scheduler

    User->>SG: create(prompt="What is AI?")
    Note over SG: status = Waiting
    SG->>Seq: create(prompt=[2061, 318, 9552, 30])
    Note over Seq: is_prefill() = true<br/>num_tokens() = 4

    Sched->>SG: schedule()
    Note over SG: status = Running
    Note over Seq: status = Running

    Sched->>Seq: prefill (process prompt)
    Note over Seq: output = [464]<br/>is_prefill() = false<br/>num_tokens() = 5

    loop decode iterations
        Sched->>Seq: decode (generate 1 token)
        Note over Seq: output grows by 1<br/>num_tokens() increments
    end

    Seq->>SG: stop token reached
    Note over SG: status = Finished
    Note over Seq: status = Finished
```
**Figure 11.7** --- Sequence lifecycle within a SequenceGroup. The sequence transitions from Waiting through Running (prefill then decode) to Finished. The group's status mirrors its sequences.
