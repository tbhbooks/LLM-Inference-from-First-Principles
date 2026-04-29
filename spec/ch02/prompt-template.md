# Chapter 2 -- LLM Prompt Template

Copy and paste this prompt into your LLM of choice to generate a working
implementation. This builds on the project created in Chapter 1.

---

## Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 2.
I already have a project directory from Chapter 1 with this structure:

  rvllm/
  ├── Cargo.toml (or equivalent project file)
  ├── src/
  │   └── main.rs (minimal placeholder)
  └── examples/
      └── ch01_inference_problem.[ext]   <-- Chapter 1 (KV cache calculator)

Add a NEW example program to this project: an architecture overview printer
that compares vLLM's production architecture with rvllm's simplified design.

TARGET LANGUAGE: [Same language as Chapter 1]

=== WHAT TO CREATE ===

Add ONE new file:

  rvllm/examples/ch02_architecture.[ext]

For Rust, also add to Cargo.toml:
  [[example]]
  name = "ch02_architecture"
  path = "examples/ch02_architecture.rs"

Do NOT modify any Chapter 1 files. Only ADD the new example.

=== PROGRAM SPECIFICATION ===

Build a CLI program that prints an educational overview of vLLM's architecture.
The program is purely text output -- no computation, no dependencies beyond
standard I/O. It prints ASCII art diagrams, tables, and explanatory text.

The program has a main() that calls these 7 functions in order:

1. print_header()
2. print_vllm_architecture()
3. print_rvllm_architecture()
4. print_comparison_table()
5. print_request_lifecycle()
6. print_key_simplifications()
7. print_footer()

=== SECTION CONTENT ===

HEADER:
- Title banner with "====" separators (80-char lines)
- Title: "Chapter 2: vLLM Architecture Overview"
- Intro text: vLLM uses six layers deep in the real implementation, rvllm
  distills this into seven flat modules that are easier to learn, build,
  and reason about.
- Close: "Let's see both side by side, then trace a request from arrival
  to response."

SECTION 1: "vLLM's Six-Layer Architecture"
- Separator: "----" (80-char line)
- Intro: a request passes through six layers, each adding a capability
- ASCII box diagram (vertical stack, top to bottom):
  1. FastAPI HTTP Server - Accepts OpenAI-compatible requests, validates params
  2. AsyncLLM - Front-end async wrapper, bridges HTTP async to engine,
     manages request queues, returns results via asyncio futures
  3. EngineCoreClient - IPC bridge using ZMQ sockets, sends requests to
     separate engine process, decouples API from engine
  4. EngineCore - The brain, runs scheduling loop: pick sequences, manage
     KV cache blocks, dispatch work to executor
  5. MultiprocExecutor - Multi-GPU process manager, spawns one worker per GPU,
     coordinates tensor-parallel execution
  6. GPUWorker + GPUModelRunner - Actual computation, loads weights, runs
     attention, executes CUDA kernels, manages GPU memory
- Each box connected by arrows (vertical lines + down arrows)
- Note between layers 3 and 4: "(ZMQ IPC)"
- Closing: "That's a lot of layers! The ZMQ IPC and multi-GPU process management
  exist for production scalability, but they obscure the core logic."

SECTION 2: "rvllm's Seven-Module Architecture"
- Separator: "----" (80-char line)
- Intro: rvllm strips away distributed-systems plumbing, exposes seven concepts
- ASCII box diagram showing two rows:
  Row 1: api/ --> engine/ --> scheduler/ --> memory/
  Row 2: tokenizer/    model/ ----(logits)----> sampling/
  With arrows showing data flow
- Module responsibility table (7 rows):
  api/       - HTTP API layer (OpenAI-compatible)
  engine/    - Inference loop orchestration
  scheduler/ - Request queuing, batching, preemption
  model/     - Forward pass (transformer layers)
  memory/    - KV cache block management (PagedAttention)
  sampling/  - Token selection (greedy, top-k, top-p)
  tokenizer/ - Text <-> tokens conversion
- Closing: "Seven modules. Each one maps to a concept you can study, build,
  and test independently. No IPC. No multi-process coordination. Just the
  core logic."

SECTION 3: "Mapping: vLLM Component -> rvllm Module -> Book Chapter"
- Separator: "----" (80-char line)
- Three-column table:
  FastAPI HTTP Server    | api/       | Ch 10: API & Streaming
  AsyncLLM              | engine/    | Ch 11: Putting It Together
  EngineCoreClient (ZMQ)| (removed)  | --
  Scheduler             | scheduler/ | Ch 5: The Scheduler
  BlockManager          | memory/    | Ch 6: Memory & PagedAttn
  MultiprocExecutor     | (removed)  | --
  GPUModelRunner        | model/     | Ch 7-8: Model & Attention
  SamplingParams        | sampling/  | Ch 9: Sampling
  (tokenizer utils)     | tokenizer/ | Ch 4: Tokenization
  (transformer layers)  | model/     | Ch 3: Transformer Primer
- Note: Two layers vanish:
  EngineCoreClient (ZMQ IPC) - single process
  MultiprocExecutor - single GPU

SECTION 4: "Request Lifecycle: A Token's Journey"
- Separator: "----" (80-char line)
- Intro: trace "Explain PagedAttention" through rvllm
- 9 steps, each with a header "Step N: NAME" and underline:

  Step 1: ARRIVE
    HTTP POST /v1/completions { "prompt": "Explain PagedAttention", ... }
    api module validates, extracts parameters

  Step 2: TOKENIZE
    tokenizer/ converts "Explain PagedAttention" -> [50872, 7873, 58662]

  Step 3: ENQUEUE
    scheduler/ adds to WAITING queue
    Show: waiting: [ Seq(id=42, tokens=[50872, 7873, 58662], blocks=[]) ]
           running: [ ... other sequences ... ]

  Step 4: SCHEDULE
    scheduler/ decides: fit in batch? enough blocks? preempt?
    Result: Seq(42) moves WAITING -> RUNNING

  Step 5: ALLOCATE
    memory/ assigns blocks: 1 block (3 tokens, block_size=16)
    Free blocks: 254 -> 253
    Block table: { 42: [block_7] }

  Step 6: EXECUTE
    model/ forward pass: embed -> N transformer layers -> logits

  Step 7: SAMPLE
    sampling/ picks from logits: [0.01, 0.03, ..., 0.42, ...]
    temperature=0 (greedy): argmax -> token 1334 ("Paged")

  Step 8: UPDATE
    engine/ appends: Seq(42).tokens = [50872, 7873, 58662, 1334]
    Stop? No (not EOS, not max_tokens)

  Step 9: RESPOND
    api/ streams: data: {"choices": [{"text": "Paged"}]}

- Loop box: Steps 4-8 repeat until EOS, max_tokens, or client disconnect.
  For max_tokens=100, loop might run 100 times.

SECTION 5: "Key Simplifications: What rvllm Drops (and Why)"
- Separator: "----" (80-char line)
- Intro: rvllm is a learning implementation, intentionally drops production features
- Two-column table (5 rows):
  ZMQ IPC (EngineCoreClient)       | Single process, direct function calls
  Multi-GPU (MultiprocExecutor)     | Single GPU, simple memory management
  CUDA graphs                       | Added later for perf, not needed for core
  Speculative decoding              | Optimization, covered later
  LoRA / adapter support            | Adds complexity, base system first
- Closing: "The goal: understand every line of code you're running."

FOOTER:
- "====" separator (80-char line)
- Title: "What's Next?"
- Teaser: Chapter 3 sets up the project skeleton with module boundaries,
  traits, types, error handling, and a CLI. No business logic yet — just
  the architectural foundation for everything that follows.
- "====" separator

=== FORMAT RULES ===

- All output to stdout
- Section separators: "----" repeated to 80 chars
- Header/footer separators: "====" repeated to 80 chars
- Box diagrams: Unicode box-drawing characters
- No ANSI colors or escape codes
- Exit code 0

=== WHAT TO PRODUCE ===

1. The new example file: examples/ch02_architecture.[ext]
2. Any project file changes needed (e.g., Cargo.toml example entry)

Do NOT recreate or modify Chapter 1 files.

After this chapter, the project directory looks like:

  rvllm/
  ├── Cargo.toml
  ├── src/
  │   └── main.rs
  └── examples/
      ├── ch01_inference_problem.[ext]   <-- Chapter 1
      └── ch02_architecture.[ext]        <-- Chapter 2 (NEW)
```
