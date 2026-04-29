# Chapter 12: Sequence Diagram

## Schedule Step — Three-Phase Algorithm

```mermaid
sequenceDiagram
    participant E as Engine
    participant S as Scheduler
    participant BA as BlockAllocator
    participant W as Waiting Queue
    participant R as Running Set
    participant Sw as Swapped Queue

    E->>S: schedule()

    Note over S: Phase 1: Check running

    loop each running request
        S->>S: blocks_for_next_token(req)
        alt needs new block
            S->>BA: can_allocate(1)?
            alt yes
                BA-->>S: true
                S->>BA: allocate()
            else no memory
                BA-->>S: false
                S->>Sw: preempt(req)
                Note over Sw: req → Swapped
            end
        end
    end

    Note over S: Phase 2: Resume swapped

    loop while swapped not empty
        S->>BA: can_allocate(blocks_needed)?
        alt yes
            BA-->>S: true
            Sw->>R: resume request
        else no
            BA-->>S: false
            Note over S: stop resuming
        end
    end

    Note over S: Phase 3: Admit waiting

    loop while waiting not empty
        S->>W: peek front
        S->>BA: can_allocate(prompt_blocks)?
        alt memory + budget OK
            BA-->>S: true
            W->>R: admit request
            S->>BA: allocate(N blocks)
        else insufficient
            BA-->>S: false
            Note over S: stop admitting
        end
    end

    S-->>E: SchedulerOutput
```
**Figure 12.6** --- The three-phase schedule() algorithm. Phase 1 ensures running requests can continue (preempting if memory is exhausted). Phase 2 tries to bring back swapped requests. Phase 3 admits new requests from the waiting queue. The BlockAllocator gates every admission.

## Preemption and Recovery

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant R as Running Set
    participant Sw as Swapped Queue
    participant BA as BlockAllocator

    Note over R: R0, R1, R2 all running<br/>R2 needs a new block

    S->>BA: can_allocate(1)?
    BA-->>S: false (0 free blocks)

    Note over S: Must preempt!<br/>Pick R2 (last admitted)

    S->>R: remove R2
    alt Swap policy
        S->>Sw: push R2 (keep blocks reserved)
        Note over Sw: R2 status = Swapped
    else Recompute policy
        S->>BA: free all R2's blocks
        Note over BA: +2 free blocks
        S->>S: push R2 to front of waiting
        Note over S: R2 status = Waiting<br/>(will redo prefill)
    end

    Note over S: Later... R0 finishes

    S->>BA: free R0's blocks
    Note over BA: blocks available

    alt R2 was swapped
        S->>Sw: pop R2
        S->>R: resume R2 (Swapped → Running)
    else R2 was recomputed
        Note over S: R2 re-admitted from waiting<br/>Full prefill again
    end
```
**Figure 12.7** --- Preemption and recovery under both policies. Swap preserves block assignments for fast resume. Recompute frees blocks immediately but requires re-doing prefill when the request is re-admitted.

## Multi-Step Lifecycle

```mermaid
sequenceDiagram
    participant User as Requests
    participant S as Scheduler
    participant GPU as GPU Batch

    User->>S: add R0, R1, R2, R3

    S->>GPU: Step 1: schedule()<br/>admit R0, R1, R2 (batch=3)
    Note over GPU: Prefill R0, R1, R2

    S->>GPU: Step 2: schedule()<br/>all decode
    Note over GPU: R0, R1, R2 each +1 token

    Note over S: R2 finishes (5 tokens)
    User->>S: add R3
    S->>GPU: Step 3: schedule()<br/>R2 done, admit R3
    Note over GPU: R0, R1 decode + R3 prefill

    Note over S: Memory pressure!
    S->>GPU: Step 4: schedule()<br/>preempt R3
    Note over GPU: R0, R1 decode only

    Note over S: R1 finishes
    S->>GPU: Step 5: schedule()<br/>R1 done, resume R3
    Note over GPU: R0 decode + R3 resumes
```
**Figure 12.8** --- Five-step lifecycle showing the full range of scheduler actions: admission, decoding, finishing, preemption, and resumption. R3's journey through Waiting, Running, Swapped, and back to Running demonstrates the four-state machine.
