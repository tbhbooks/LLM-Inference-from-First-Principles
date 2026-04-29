# Chapter 2: Component Diagrams

## vLLM's Six-Layer Architecture

A request passes through six layers in production vLLM. Each layer adds a capability (async handling, IPC, scheduling, multi-GPU, execution):

```mermaid
graph TD
    A["FastAPI HTTP Server<br/><i>Accepts OpenAI-compatible requests, validates parameters</i>"]
    B["AsyncLLM<br/><i>Front-end async wrapper. Bridges HTTP async world to engine.<br/>Manages request queues, returns results via asyncio futures.</i>"]
    C["EngineCoreClient<br/><i>IPC bridge using ZMQ sockets. Sends requests to a separate<br/>engine process and receives outputs. Decouples API from engine.</i>"]
    D["EngineCore<br/><i>The brain. Runs the scheduling loop:<br/>1. Pick which sequences to run (Scheduler)<br/>2. Manage KV cache blocks (BlockManager)<br/>3. Dispatch work to executor</i>"]
    E["MultiprocExecutor<br/><i>Multi-GPU process manager. Spawns one worker per GPU.<br/>Coordinates tensor-parallel execution across devices.</i>"]
    F["GPUWorker + GPUModelRunner<br/><i>The actual computation. Loads model weights, runs attention,<br/>executes CUDA kernels, manages GPU memory.</i>"]

    A --> B
    B --> C
    C -->|"ZMQ IPC"| D
    D --> E
    E --> F

```

**Legend:** Blue = API/async layers. Orange = distributed plumbing (removed in rvllm). Green = core logic.

## rvllm's Seven-Module Architecture

rvllm strips away the distributed-systems plumbing and exposes the seven concepts you need to understand LLM inference:

```mermaid
graph LR
    API["api/<br/><i>HTTP API (OpenAI)</i>"]
    ENGINE["engine/<br/><i>Inference loop</i>"]
    SCHED["scheduler/<br/><i>Queuing & batching</i>"]
    MEM["memory/<br/><i>KV cache blocks</i>"]
    TOK["tokenizer/<br/><i>Text <-> tokens</i>"]
    MODEL["model/<br/><i>Forward pass</i>"]
    SAMP["sampling/<br/><i>Token selection</i>"]

    API --> ENGINE
    ENGINE --> SCHED
    SCHED --> MEM
    ENGINE --> MODEL
    MODEL -->|"logits"| SAMP
    MEM --> SAMP

```

## Mapping: vLLM Layers to rvllm Modules

```mermaid
graph LR
    subgraph vLLM["vLLM (6 layers)"]
        V1["FastAPI HTTP Server"]
        V2["AsyncLLM"]
        V3["EngineCoreClient (ZMQ)"]
        V4["EngineCore (Scheduler + BlockManager)"]
        V5["MultiprocExecutor"]
        V6["GPUWorker + GPUModelRunner"]
    end

    subgraph rvllm["rvllm (7 modules)"]
        R1["api/"]
        R2["engine/"]
        R3["scheduler/"]
        R4["memory/"]
        R5["model/"]
        R6["sampling/"]
        R7["tokenizer/"]
    end

    V1 -.-> R1
    V2 -.-> R2
    V3 -.-x|"removed"| R2
    V4 -.-> R3
    V4 -.-> R4
    V5 -.-x|"removed"| R5
    V6 -.-> R5
    V6 -.-> R6

```

Two vLLM layers vanish entirely in rvllm:
- **EngineCoreClient (ZMQ IPC)** -- we run in a single process
- **MultiprocExecutor** -- we target a single GPU
