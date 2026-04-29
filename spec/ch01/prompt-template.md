# Chapter 1 -- LLM Prompt Template

Copy and paste this prompt into your LLM of choice to generate a working
implementation. This is the first chapter — it creates the project directory
and the first example program.

---

## Prompt

```
I am building an LLM inference engine called "rvllm" as a learning project.
This is Chapter 1. Create the project directory and the first example program:
a KV cache memory calculator that shows why LLM serving is memory-bound.

TARGET LANGUAGE: [Rust / Python / Go / your choice]

=== STEP 1: CREATE THE PROJECT ===

Create a new project called "rvllm" with this structure:

  rvllm/
  ├── examples/
  │   └── ch01_inference_problem.[ext]    <-- the program for this chapter
  └── [language-specific project files]

For Rust:
  cargo init rvllm
  The ch01 program goes in examples/ch01_inference_problem.rs
  Add to Cargo.toml:
    [[example]]
    name = "ch01_inference_problem"
    path = "examples/ch01_inference_problem.rs"

For Python:
  mkdir -p rvllm/examples
  The ch01 program goes in examples/ch01_inference_problem.py

The project directory will grow over subsequent chapters. Future chapters
will add source modules and more examples into this same directory.

=== STEP 2: IMPLEMENT THE KV CACHE CALCULATOR ===

This program demonstrates why LLM serving is memory-bound by computing
KV cache sizes for real transformer models. No model loading, no GPU — just math.

== Data Structure: ModelConfig ==

A struct/class with these fields:

    name: string          — Human-readable model name
    num_layers: int       — Number of transformer layers
    num_heads: int        — Number of KV attention heads
    head_dim: int         — Dimension per attention head
    dtype_bytes: int      — Bytes per element (2 for float16/bfloat16)

== Methods on ModelConfig ==

Each builds on the previous (smallest unit → largest):

    kv_vector_bytes()
        = num_heads * head_dim * dtype_bytes

    kv_per_token_per_layer()
        = 2 * kv_vector_bytes()

    kv_per_token()
        = kv_per_token_per_layer() * num_layers

    kv_for_sequence(seq_len)
        = kv_per_token() * seq_len

    max_concurrent_sequences(seq_len, memory_bytes)
        = memory_bytes / kv_for_sequence(seq_len)    [integer division]
        (return 0 if kv_for_sequence is 0)

== Helper: format_bytes(bytes) -> string ==

Convert bytes to human-readable using binary units (1024-based):
- >= 1 GB (1024^3): format as "X.XX GB"
- >= 1 MB (1024^2): format as "X.XX MB"
- >= 1 KB (1024):   format as "X.XX KB"
- else:             format as "X B"
Always 2 decimal places for KB/MB/GB.

== Preset Models ==

Create a factory function returning exactly these three:

    GPT-2 (124M):  num_layers=12,  num_heads=12, head_dim=64,  dtype_bytes=2
    LLaMA-7B:      num_layers=32,  num_heads=32, head_dim=128, dtype_bytes=2
    LLaMA-70B:     num_layers=80,  num_heads=64, head_dim=128, dtype_bytes=2

== Constants ==

    gpu_memory = 10 * 1024 * 1024 * 1024   (10 GB = 10,737,418,240 bytes)
    concurrent_counts = [1, 2, 4, 8, 16, 32]

== Output Format ==

The program prints 5 sections. Each section header is printed with:
- blank line
- 78 '=' characters
- "  {TITLE}" (2-space indent)
- 78 '=' characters
- blank line

=== PART 1: KV Cache Memory — Where Does It All Go? ===

Print 2 introductory lines, then a table:

  Model              Layers    Heads  HeadDim      Per Token  Per Tok/Layer
  ------------------------------------------------------------------------
  GPT-2 (124M)           12       12       64       36.00 KB        3.00 KB
  LLaMA-7B               32       32      128      512.00 KB       16.00 KB
  LLaMA-70B              80       64      128        2.50 MB       32.00 KB

Column widths: Model=16 left-aligned, Layers/Heads/HeadDim=8 right-aligned,
Per Token/Per Tok/Layer=14 right-aligned. Separator: 72 dashes, 2-space indent.

=== PART 2: KV Cache vs Sequence Length ===

  Model                 1024 tokens      4096 tokens
  --------------------------------------------------
  GPT-2 (124M)             36.00 MB        144.00 MB
  LLaMA-7B                512.00 MB          2.00 GB
  LLaMA-70B                 2.50 GB         10.00 GB

Column widths: Model=16 left, others=16 right. Separator: 50 dashes.

=== PART 3: The Concurrency Ceiling (10 GB GPU Memory) ===

Print "Given 10.00 GB of free GPU memory for KV cache," and second line.
Then table:

  Model                 Max @ 1024 tokens      Max @ 4096 tokens
  --------------------------------------------------------------
  GPT-2 (124M)                    284 seq                 71 seq
  LLaMA-7B                         20 seq                  5 seq
  LLaMA-70B                         4 seq                  1 seq

Column widths: Model=16 left, each max column=22 right. The value is
right-aligned in 18 chars followed by " seq".
Separator: 62 dashes.
After table print the "Notice" lines about LLaMA-70B.

=== PART 4: The Memory Wall — Scaling Concurrent Requests ===

For each model, for each count in [1, 2, 4, 8, 16, 32]:
- needed = kv_for_sequence(1024) * count
- needed_gb = needed / (1024^3) as float
- If count <= max_concurrent_sequences(1024, gpu_memory): print "{needed_gb:.1f}GB"
  in 9-char right-aligned field
- Else: print "OOM!!" in 9-char right-aligned field (right-align "OOM" in 7 chars then "!!")

Header row: Model=16 left, then each "{n}req" in 9-char right field.
Separator: 72 dashes. After table: "OOM!! = Out of memory..."

=== PART 5: Prefill vs Decode — Two Very Different Phases ===

Print static text explaining:
- PREFILL: parallel prompt processing, compute-bound
- DECODE: sequential generation, memory-bound
- ASCII box: "LLM serving is fundamentally a MEMORY problem."

=== Closing ===

After Part 5, print a separator with:
  "  Chapter 1 complete. Next: vLLM architecture overview (ch02)"
and another separator.

=== VALIDATION ===

Your output should contain these exact strings:
- "36.00 KB" (GPT-2 per token)
- "512.00 KB" (LLaMA-7B per token)
- "2.50 MB" (LLaMA-70B per token)
- "284 seq" and "71 seq" (GPT-2 concurrency)
- "20 seq" and "5 seq" (LLaMA-7B concurrency)
- "4 seq" and "1 seq" (LLaMA-70B concurrency)
- "OOM!!" (memory wall section)
- "PART 1" through "PART 5" in section headers

=== WHAT TO PRODUCE ===

1. The project directory setup commands (or project files like Cargo.toml)
2. The complete source file for examples/ch01_inference_problem.[ext]

After this chapter, the project directory will look like:

  rvllm/
  ├── Cargo.toml (or equivalent)
  ├── src/
  │   └── main.rs (or lib, can be minimal placeholder)
  └── examples/
      └── ch01_inference_problem.[ext]
```
