# Building vLLM from Scratch — Spec-Driven Edition

A practical developer book that teaches LLM inference serving through **spec-driven development**. Instead of copying code, you build it yourself — guided by architecture diagrams, interface specs, and validation tests.

Written for **both human readers and LLM agents**. Rich enough to read. Structured enough to act on.

## How This Book Works

Each chapter follows a rhythm — but with chapter-specific, engaging section headers (not generic labels):

1. **Opening** — A vivid scenario that pulls you into the problem
2. **Tension** — What goes wrong, what's missing, why this matters
3. **Insight** — The key idea, with diagrams and architecture
4. **The Spec** — What you need to build (points to `../spec/chNN/`)
5. **See It** — Expected output, before/after, "the moment it works"
6. **Try It** — Experiments and extensions
7. **What's Next** — What breaks next, teasing the next chapter

## What You'll Build

An LLM inference engine — the same core ideas as vLLM, built from scratch in your language of choice.

| Part | Chapters | You'll Have |
|------|----------|-------------|
| I: Foundations | 00-02 | Understanding of the problem and architecture |
| II: MVP | 03-08 | Working single-request inference from GPT-2 |
| III: Core vLLM | 09-14 | PagedAttention, continuous batching, scheduler |
| IV: Production | 15-19 | API server, prefix caching, speculative decoding |
| V: Further | 20 | Research landscape and next steps |

## Getting Started

1. Read a chapter in `book2/`
2. Review the spec in `spec/chNN/`
3. Build your implementation (or use `spec/chNN/prompt-template.md` with an LLM)
4. Validate: `cd spec/chNN/validation && pytest`

## Running Example

Throughout the book, we trace a single prompt through the system:

> "What is AI?" → token IDs: `[2061, 318, 9552, 30]`

These four tokens will flow through tokenizers, embedding tables, attention heads, KV caches, block tables, and schedulers. By the end, you'll know every step of their journey.

## Reference Implementation

A reference implementation (in Rust) lives in `rvllm/`. Git tags (`ch00`, `ch01`, ...) snapshot the code at each chapter. Use it if you get stuck — but try the spec-first approach first.

## Prerequisites

- A programming language you're comfortable with
- Python 3.10+ with pytest (for validation tests)
- For Chapter 4+: a machine that can run GPT-2 (CPU works, GPU faster)
- Curiosity about how LLM inference actually works
