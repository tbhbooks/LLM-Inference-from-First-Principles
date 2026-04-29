# Chapter 13: Component Diagram

## Engine — The Orchestrator

```mermaid
classDiagram
    class Engine {
        -config: EngineConfig
        -scheduler: Scheduler
        -model: Model
        -tokenizer: TokenizerBackend
        -sampler: Sampler
        -block_allocator: BlockAllocator
        -sequences: Map~RequestId, SequenceData~
        +add_request(id, prompt, max_tokens)
        +step() StepOutput
        +run(max_steps) List~CompletedRequest~
    }

    class Scheduler {
        <<trait / ch12>>
        +add_sequence(id, blocks_needed)
        +schedule() SchedulerOutput
        +finish_sequence(id)
    }

    class Model {
        <<trait / ch05-06>>
        +forward(token_ids, positions, block_tables) Logits
    }

    class TokenizerBackend {
        <<trait / ch04>>
        +encode(text) List~TokenId~
        +decode(token_ids) String
        +eos_token_id() TokenId
    }

    class Sampler {
        <<trait / ch07>>
        +sample(logits) List~TokenId~
    }

    class BlockAllocator {
        <<trait / ch10>>
        +allocate() BlockId
        +free(block_id)
        +num_free_blocks() int
    }

    class SequenceData {
        +request_id: RequestId
        +token_ids: List~int~
        +prompt_len: int
        +status: SequenceStatus
        +block_table: BlockTable
        +max_tokens: int
    }

    class StepOutput {
        +completed: List~CompletedRequest~
        +num_running: int
        +num_waiting: int
        +num_scheduled: int
        +num_tokens_generated: int
    }

    Engine --> Scheduler : asks "who runs next?"
    Engine --> Model : calls forward()
    Engine --> TokenizerBackend : encode/decode
    Engine --> Sampler : picks next token
    Engine --> BlockAllocator : allocate/free blocks
    Engine --> SequenceData : tracks all sequences
    Engine ..> StepOutput : returns per step
```
**Figure 13.1** — Engine class structure. The Engine holds references to five trait-based components (scheduler, model, tokenizer, sampler, block allocator) and a map of sequence data. The `step()` method orchestrates them all.

## The step() Pipeline

```mermaid
flowchart LR
    S["1. Schedule<br/>(ch12)"] --> A["2. Allocate<br/>blocks (ch10)"]
    A --> P["3. Prepare<br/>inputs"]
    P --> F["4. Forward<br/>(ch05-06)"]
    F --> SM["5. Sample<br/>(ch07)"]
    SM --> U["6. Update<br/>sequences"]
    U -->|"more to do"| S
    U -->|"all done"| D["Return<br/>completed"]

```
**Figure 13.2** — The six phases of `step()`. Each iteration flows left to right: schedule, allocate, prepare, forward, sample, update. If sequences remain, loop back to schedule.

## Request Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> Waiting : add_request()
    Waiting --> Running : scheduler admits
    Running --> Running : step() generates token
    Running --> Finished : EOS or max_tokens
    Finished --> [*] : blocks freed
```
**Figure 13.3** — Sequence lifecycle. A request starts as Waiting, moves to Running when the scheduler admits it, stays Running while generating tokens, and transitions to Finished when a stop condition is met.
