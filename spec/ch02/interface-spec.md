# Chapter 2: Interface Specification

## Overview

The Chapter 2 program is a **pure text output** CLI. It performs no computation -- it prints structured ASCII art and text that teaches vLLM's architecture, rvllm's simplification, and the request lifecycle. Any language can implement it.

## Program Structure

The program consists of a `main` entry point that calls 6 functions in order:

```
main()
  1. print_header()
  2. print_vllm_architecture()
  3. print_rvllm_architecture()
  4. print_comparison_table()
  5. print_request_lifecycle()
  6. print_key_simplifications()
  7. print_footer()
```

**Note:** Despite having 7 functions, the program produces 6 distinct content sections (header and footer are framing, not sections). The 6 section headers are numbered 1-5 plus the footer.

## Function Specifications

### 1. `print_header()`

Prints a title banner and introductory paragraph.

**Must contain:**
- A `====` separator line (80 chars)
- Title: `Chapter 2: vLLM Architecture Overview`
- Introductory text mentioning: "six layers deep", "seven flat modules"
- Closing teaser: "Let's see both side by side, then trace a request"

### 2. `print_vllm_architecture()`

Prints section "1. vLLM's Six-Layer Architecture" with an ASCII box diagram.

**Section header:** `1. vLLM's Six-Layer Architecture`

**Must list these 6 components (in order, top to bottom):**

| Layer | Component | Description |
|-------|-----------|-------------|
| 1 | FastAPI HTTP Server | Accepts OpenAI-compatible requests, validates parameters |
| 2 | AsyncLLM | Front-end async wrapper. Bridges HTTP async world to engine. Manages request queues, returns results via asyncio futures. |
| 3 | EngineCoreClient | IPC bridge using ZMQ sockets. Sends requests to a separate engine process and receives outputs. Decouples API from engine. |
| 4 | EngineCore | The brain. Runs the scheduling loop: (1) Pick which sequences to run, (2) Manage KV cache blocks, (3) Dispatch work to executor |
| 5 | MultiprocExecutor | Multi-GPU process manager. Spawns one worker per GPU. Coordinates tensor-parallel execution across devices. |
| 6 | GPUWorker + GPUModelRunner | The actual computation. Loads model weights, runs attention, executes CUDA kernels, manages GPU memory. |

**Format:** Each component in a box made of Unicode box-drawing characters, connected by arrows (`|`, `v`).

**Closing remark:** Mention that ZMQ IPC and multi-GPU process management obscure the core logic.

### 3. `print_rvllm_architecture()`

Prints section "2. rvllm's Seven-Module Architecture" with a flow diagram and a module table.

**Section header:** `2. rvllm's Seven-Module Architecture`

**Must list these 7 modules:**

| Module | Responsibility |
|--------|---------------|
| api/ | HTTP API layer (OpenAI-compatible) |
| engine/ | Inference loop orchestration |
| scheduler/ | Request queuing, batching, preemption |
| model/ | Forward pass (transformer layers) |
| memory/ | KV cache block management (PagedAttention) |
| sampling/ | Token selection (greedy, top-k, top-p) |
| tokenizer/ | Text <-> tokens conversion |

**Format:** Two rows of boxes showing data flow (api -> engine -> scheduler -> memory on top row; tokenizer, model, sampling on bottom row). Followed by a table listing all 7 modules and responsibilities.

**Closing remark:** "Seven modules. Each one maps to a concept you can study, build, and test independently."

### 4. `print_comparison_table()`

Prints section "3. Mapping: vLLM Component -> rvllm Module -> Book Chapter".

**Section header:** `3. Mapping: vLLM Component -> rvllm Module -> Book Chapter`

**Must contain this mapping table:**

| vLLM Component | rvllm Module | Book Chapter |
|----------------|-------------|--------------|
| FastAPI HTTP Server | api/ | Ch 10: API & Streaming |
| AsyncLLM | engine/ | Ch 11: Putting It Together |
| EngineCoreClient (ZMQ) | (removed) | -- |
| Scheduler | scheduler/ | Ch 5: The Scheduler |
| BlockManager | memory/ | Ch 6: Memory & PagedAttn |
| MultiprocExecutor | (removed) | -- |
| GPUModelRunner | model/ | Ch 7-8: Model & Attention |
| SamplingParams | sampling/ | Ch 9: Sampling |
| (tokenizer utils) | tokenizer/ | Ch 4: Tokenization |
| (transformer layers) | model/ | Ch 3: Transformer Primer |

**Closing remark:** Explicitly state the two removed layers:
- EngineCoreClient (ZMQ IPC) -- single process
- MultiprocExecutor -- single GPU

### 5. `print_request_lifecycle()`

Prints section "4. Request Lifecycle: A Token's Journey" with 9 detailed steps.

**Section header:** `4. Request Lifecycle: A Token's Journey`

**Opening:** "Let's trace a single request -- 'Explain PagedAttention' -- through rvllm."

**Must include all 9 steps with these exact step names and details:**

| Step | Name | Details |
|------|------|---------|
| 1 | ARRIVE | HTTP POST hits api/ module. POST /v1/completions with prompt "Explain PagedAttention" |
| 2 | TOKENIZE | tokenizer/ converts "Explain PagedAttention" -> [50872, 7873, 58662] |
| 3 | ENQUEUE | scheduler/ adds sequence to WAITING queue. Show waiting/running queues. |
| 4 | SCHEDULE | scheduler/ decides what runs. Checks batch capacity, free blocks, preemption. Seq moves WAITING -> RUNNING. |
| 5 | ALLOCATE | memory/ assigns KV cache blocks. 1 block for 3 tokens (block_size=16). Free blocks 254->253. Block table shown. |
| 6 | EXECUTE | model/ runs forward pass. Embed, transformer layers, produce logits. |
| 7 | SAMPLE | sampling/ picks next token. Show logits array, greedy argmax -> token 1334 ("Paged"). |
| 8 | UPDATE | engine/ appends token, checks stop criteria. tokens now [..., 1334]. Not EOS, not max_tokens. |
| 9 | RESPOND | api/ streams token back. Show SSE data format. |

**Loop box:** After all 9 steps, include a box stating steps 4-8 repeat until EOS, max_tokens, or client disconnect. Mention that for max_tokens=100, the loop might run 100 times.

### 6. `print_key_simplifications()`

Prints section "5. Key Simplifications: What rvllm Drops (and Why)".

**Section header:** `5. Key Simplifications: What rvllm Drops (and Why)`

**Must list these 5 dropped features:**

| Dropped Feature | Why It's OK |
|----------------|-------------|
| ZMQ IPC (EngineCoreClient) | Single process -- direct function calls replace socket messaging. |
| Multi-GPU support (MultiprocExecutor) | Single GPU keeps memory management simple. No tensor parallelism yet. |
| CUDA graphs | We'll add these later for perf. Not needed to understand the core. |
| Speculative decoding | An optimization on top of the base loop. Covered in a later chapter. |
| LoRA / adapter support | Adds memory management complexity. The base system comes first. |

**Closing remark:** "The goal: understand every line of code you're running. No magic."

### 7. `print_footer()`

Prints a closing section with `====` separators.

**Must contain:**
- Title: `What's Next?`
- Teaser for Chapter 3 (transformer / model/ module / forward pass)
- Tension-building question about KV cache memory growth
- Mention PagedAttention and Chapter 6

## Output Format Rules

1. All output goes to stdout
2. Section separators use `----` (80-char lines)
3. Header/footer separators use `====` (80-char lines)
4. Box diagrams use Unicode box-drawing: `+---+`, `|`, arrows
5. Tables use Unicode box-drawing: `+---+---+` style
6. No color codes or ANSI escape sequences
7. Lines should not exceed 80 characters where possible
8. Exit code 0 on success
