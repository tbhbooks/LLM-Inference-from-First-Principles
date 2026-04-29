# Chapter 12: Component Diagram

## SequenceStatus State Machine (Updated)

```mermaid
stateDiagram-v2
    [*] --> Waiting : request arrives
    Waiting --> Running : scheduler admits<br/>(memory available)
    Running --> Finished : stop token or max length
    Running --> Swapped : preempted<br/>(memory pressure)
    Swapped --> Running : resumed<br/>(memory freed)
    Finished --> [*] : resources freed
```
**Figure 12.1** --- Updated SequenceStatus with Swapped state. A running request can be preempted to Swapped when memory runs out, then resumed when blocks become available. Chapter 11's three-state machine becomes four.

## Three-Queue Architecture

```mermaid
graph LR
    subgraph "Scheduler Queues"
        W["Waiting Queue<br/>(FIFO by arrival)"]
        R["Running Set<br/>(active on GPU)"]
        S["Swapped Queue<br/>(preempted, FIFO)"]
    end

    W -->|"admit<br/>(memory OK)"| R
    R -->|"preempt<br/>(no memory)"| S
    S -->|"resume<br/>(memory freed)"| R
    R -->|"finish"| F["Done<br/>(blocks freed)"]

```
**Figure 12.2** --- The scheduler's three queues. Requests flow from Waiting to Running when admitted. Under memory pressure, Running requests can be preempted to Swapped. When memory frees up, Swapped requests resume.

## Scheduler Class Structure

```mermaid
classDiagram
    class Scheduler {
        <<trait>>
        +schedule() SchedulerOutput
        +add_request(group)
        +notify_finished(group_id)
        +num_waiting() int
        +num_running() int
        +num_swapped() int
    }

    class FcfsScheduler {
        -waiting: Queue~SequenceGroup~
        -running: List~SequenceGroup~
        -swapped: Queue~SequenceGroup~
        -config: SchedulerConfig
        -block_allocator: BlockAllocator
        +schedule() SchedulerOutput
        +add_request(group)
        +notify_finished(group_id)
        -preempt(group, output)
        -blocks_for_next_token(group) int
    }

    class SchedulerConfig {
        +max_num_seqs: int
        +max_num_batched_tokens: int
        +block_size: int
        +preemption_policy: PreemptionPolicy
    }

    class SchedulerOutput {
        +new_requests: List~SequenceGroup~
        +running_requests: List~SequenceGroup~
        +preempted_ids: List~GroupId~
        +num_prefill_tokens: int
        +num_decode_tokens: int
    }

    class PreemptionPolicy {
        <<enum>>
        Swap
        Recompute
    }

    Scheduler <|.. FcfsScheduler : implements
    FcfsScheduler --> SchedulerConfig : uses
    FcfsScheduler --> SchedulerOutput : produces
    FcfsScheduler --> BlockAllocator : checks memory
    SchedulerConfig --> PreemptionPolicy : has
```
**Figure 12.3** --- Scheduler class hierarchy. The Scheduler trait defines the interface. FcfsScheduler implements it with three queues and FCFS ordering. It consults the BlockAllocator before admitting requests.

## Schedule Algorithm Flow

```mermaid
flowchart TD
    Start["schedule() called"] --> P1["Phase 1:<br/>Check running requests"]
    P1 --> P1Check{"Each running seq:<br/>needs new block?"}
    P1Check -->|"No"| P1Keep["Keep in running<br/>budget -= 1 decode token"]
    P1Check -->|"Yes"| P1Mem{"can_allocate(1)?"}
    P1Mem -->|"Yes"| P1Alloc["Allocate block<br/>keep in running"]
    P1Mem -->|"No"| P1Pre["PREEMPT<br/>(swap or recompute)"]

    P1Keep --> P2
    P1Alloc --> P2
    P1Pre --> P2

    P2["Phase 2:<br/>Resume swapped"] --> P2Check{"swapped queue<br/>not empty?"}
    P2Check -->|"Yes"| P2Mem{"memory + budget<br/>available?"}
    P2Check -->|"No"| P3
    P2Mem -->|"Yes"| P2Resume["Resume request<br/>Swapped → Running"]
    P2Mem -->|"No"| P3
    P2Resume --> P2Check

    P3["Phase 3:<br/>Admit waiting"] --> P3Check{"waiting queue<br/>not empty?"}
    P3Check -->|"Yes"| P3Mem{"can_allocate(N)<br/>+ token budget?"}
    P3Check -->|"No"| Done
    P3Mem -->|"Yes"| P3Admit["Admit request<br/>Waiting → Running"]
    P3Mem -->|"No"| Done
    P3Admit --> P3Check

    Done["Return SchedulerOutput"]

```
**Figure 12.4** --- The schedule() algorithm in three phases. Phase 1 ensures running requests can continue (preempting if not). Phase 2 tries to resume swapped requests. Phase 3 admits new requests from the waiting queue. Each phase respects both memory and batch-size budgets.

## Memory-Aware Admission

```mermaid
graph TB
    subgraph "BlockAllocator (ch10)"
        Pool["Block Pool<br/>10 blocks total"]
        Free["Free: 3 blocks"]
        Used["Used: 7 blocks"]
    end

    subgraph "Scheduler Decision"
        Check["R4 needs 2 blocks<br/>can_allocate(2)?"]
        Yes["Yes → Admit R4"]
        No["No → R4 stays in waiting"]
    end

    Pool --> Free
    Pool --> Used
    Free -->|"3 >= 2"| Yes

```
**Figure 12.5** --- Memory-aware admission. The scheduler asks the BlockAllocator if enough free blocks exist before admitting a request. This prevents overcommitting GPU memory.
