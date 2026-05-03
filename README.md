# The Builder's Handbook (TBH): LLM Inference from First Principles

> **⚠️ Work in Progress** — This book is actively being written. Chapters may change.

**Learn How LLM Inference Works at Scale — From the inside Out.**

You send a prompt to ChatGPT, Claude, or a local model. Tokens stream back one by one. Behind the scenes, an inference engine is managing memory, batching requests, and deciding what to compute next.

Most developers never look inside that machine. This book opens it up — and optionally guides you through building one from scratch. Same core ideas behind [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [TGI](https://github.com/huggingface/text-generation-inference), and every production inference engine. Read to learn, or build to internalize — either way, you'll understand every layer.

## Two Ways to Read

**Read only.** Start at Chapter 1. Every concept is explained with diagrams, traces, and full output — no setup required to follow along.

**Build as you read.** Start at Chapter 0 to set up your environment. Each chapter has a spec and validation tests. You build, you run, you see it work. Specs are language-agnostic — bring Rust, Python, Go, or whatever you prefer.

## Agent Quick Start

If you are using Cursor, Claude Code, Codex, OpenClaw, or another coding agent:

1. Open this repo in your coding agent.
2. Ask it to read [`AGENTS.md`](AGENTS.md) first.
3. Let it inspect `agents/manifest.json`, [`BOOK_CONTEXT.md`](BOOK_CONTEXT.md), and [Chapter 0](chapters/ch00-setup.md).
4. If it offers to install [`tbhbooks-agent-kit`](https://github.com/tbhbooks/tbhbooks-agent-kit), approve only after it shows the pinned version and release URL.
5. Ask: `Start with Chapter 0 setup for this TBH book, create the rvllm workspace, then continue to Chapter 1 in guided build mode.`

The default build target for this book is `rvllm` in Rust. Validation is subprocess-driven: the agent should help you build a chapter binary, set the matching `RVLLM_CHNN_BIN` environment variable, and run the chapter tests in `spec/chNN/validation/`.

## Table of Contents

### Getting Started

| Ch | Title | What You Build |
|----|-------|----------------|
| -- | Preface | Why this book exists, the roadmap, running example |
| 00 | Setup (Optional) | Prerequisites, workflow, validation harness |

### Part I: Foundations

| Ch | Title | What You Build |
|----|-------|----------------|
| 01 | The LLM Inference Problem | Why inference is hard — latency, memory, throughput |
| 02 | vLLM Architecture Overview | Mental model of the full system |
| 03 | Setting Up the Project | Project structure, dependencies, smoke test |

### Part II: MVP — Single-Request Inference

| Ch | Title | What You Build |
|----|-------|----------------|
| 04 | GPT-2 from Scratch | Load and run a real model end-to-end |
| 05 | The Building Blocks | Layers, attention, MLP — the Transformer stack |
| 06 | Where the Model Learns to Look Back | KV cache for autoregressive generation |
| 07 | The Skeleton Speaks | First working generation — prompt in, text out |
| 08 | Fit and Finish | Tokenizer integration, clean output, greedy decode |

### Part III: Core vLLM — Memory & Scheduling

| Ch | Title | What You Build |
|----|-------|----------------|
| 09 | The Memory Problem | Why naive KV caching breaks at scale |
| 10 | Paged Attention | Block-based KV cache — virtual memory for attention |
| 11 | Continuous Batching | Serve multiple requests without wasting compute |
| 12 | The Scheduler | Priority, preemption, fairness across requests |
| 13 | The Engine Loop | Orchestrating scheduler → model → output |

### Part IV: Production

| Ch | Title | What You Build |
|----|-------|----------------|
| 14 | Sampling Strategies | Temperature, top-k, top-p, repetition penalty |
| 15 | Building the API Server | OpenAI-compatible HTTP + streaming |
| 16 | Prefix Caching | Reuse KV blocks across prompts with shared prefixes |
| 17 | Speculative Decoding | Draft model + verification for faster generation |
| 18 | Structured Output | Constrained decoding — JSON, grammar, schema |
| 19 | Parallelism | Tensor and pipeline parallelism across devices |

### Part V: Further

| Ch | Title | What You Build |
|----|-------|----------------|
| 20 | Where to Go from Here | Research landscape, open problems, next steps |

## Running Example

Throughout the book, we trace a single prompt through the system:

> "What is AI?" → token IDs: `[2061, 318, 9552, 30]`

These four tokens flow through tokenizers, embedding tables, attention heads, KV caches, block tables, and schedulers. By the end, you'll know every step of their journey.

## How to Use This Book

1. **Read the chapter** — understand the concept and why it matters.
2. **Read the spec** — each chapter has a spec in `spec/chNN/` with interface contracts and expected behavior.
3. **Build it** — implement the spec in your language. No code to copy. The spec is all you need.
4. **Validate** — run the validation tests in `spec/chNN/validation/` to confirm your implementation works.
5. **Move on** — each chapter builds on the last. Your inference engine grows incrementally.

Using Claude Code? Install The Builder's Handbook (TBH) plugin for a guided build-along experience — specs, hints, validation, and progress tracking right inside your terminal:

```bash
/plugin marketplace add tbhbooks/tbh-skill
/plugin install tbh@the-builders-handbook
/tbh:setup
```

Using Cursor, Codex, OpenClaw, or another coding agent? Follow the [Agent Quick Start](#agent-quick-start). Runtime shim files (`CURSOR.md`, `CODEX.md`, `CLAUDE.md`) point to [`AGENTS.md`](AGENTS.md); other agents should fall back to it directly.

## Specs Define "Done"

Each chapter's spec lives in `spec/chNN/`:

```
spec/chNN/
├── prompt-template.md      What to implement (language-agnostic)
├── interface-spec.md       API contracts and types
├── expected-output.txt     What the program should produce
├── component-diagram.md    Architecture diagram
├── sequence-diagram.md     Data flow diagram
└── validation/
    └── test_chNN.py        Automated tests your code must pass
```

## Prerequisites

- A programming language you're comfortable with
- Python 3.10+ with pytest (for validation tests)
- Rust if you want the default `rvllm` build path
- For Chapter 4+: a machine that can run GPT-2 (CPU works, GPU faster)
- Curiosity about how LLM inference actually works

## License

Copyright (c) 2026 Rushit Patel. All rights reserved. See [LICENSE](./LICENSE).

---

*"tbh, the spec is all you need."*
