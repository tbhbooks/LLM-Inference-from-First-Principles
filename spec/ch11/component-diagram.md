# Chapter 11: Component Diagram

## SequenceStatus State Machine

```mermaid
stateDiagram-v2
    [*] --> Waiting : request arrives
    Waiting --> Running : scheduler picks it
    Running --> Finished : stop token or max length
    Finished --> [*] : resources freed
```
**Figure 11.1** --- SequenceStatus lifecycle. A request moves through three states, never backward. Chapter 12 adds a Swapped state for preemption.

## SequenceGroup Structure

```mermaid
classDiagram
    class SequenceStatus {
        <<enum>>
        Waiting
        Running
        Finished
    }

    class SequenceGroup {
        +GroupId group_id
        +list~Sequence~ sequences
        +float arrival_time
        +SequenceStatus status
        +is_finished() bool
    }

    class Sequence {
        +SeqId seq_id
        +list~int~ prompt_tokens
        +list~int~ output_tokens
        +SequenceStatus status
        +num_tokens() int
        +is_prefill() bool
    }

    SequenceGroup --> SequenceStatus : has
    SequenceGroup --> Sequence : contains 1..*
    Sequence --> SequenceStatus : has
```
**Figure 11.2** --- SequenceGroup and Sequence types. A group contains one or more sequences (beam search produces multiple). Each tracks its own status and token counts.

## Static vs Continuous Batching

```mermaid
graph TB
    subgraph "Static Batching"
        S1["Batch 1: R0, R1, R2, R3<br/>Run 10 iterations<br/>(wait for slowest)"]
        S2["Batch 2: R4, R5, R6, R7<br/>Run 7 iterations<br/>(wait for slowest)"]
        S1 --> S2
        S3["R2 done at iter 2<br/>IDLES for 8 iterations"]
        S1 -.-> S3
    end

    subgraph "Continuous Batching"
        C1["Iter 1: R0, R1, R2, R3"]
        C2["Iter 2: R0, R1, R4, R3<br/>R2 done → R4 enters"]
        C3["Iter 3: R5, R1, R4, R3<br/>R0 done → R5 enters"]
        C4["...batch stays full..."]
        C1 --> C2 --> C3 --> C4
    end

```
**Figure 11.3** --- Static batching runs each batch to completion, leaving idle slots. Continuous batching swaps requests every iteration, keeping the batch full.

## Mixed Batch: Prefill + Decode

```mermaid
graph LR
    subgraph "GPU Iteration 3"
        D1["Slot 0: R0<br/>decode (token 3/3)"]
        D2["Slot 1: R1<br/>decode (token 3/8)"]
        P1["Slot 2: R4<br/>PREFILL (10 tokens)"]
        D3["Slot 3: R3<br/>decode (token 3/10)"]
    end

```
**Figure 11.4** --- A mixed batch at iteration 3. Three requests are in decode phase (generating one token each), while R4 just entered and runs prefill (processing all prompt tokens at once). The scheduler manages both in the same GPU pass.
