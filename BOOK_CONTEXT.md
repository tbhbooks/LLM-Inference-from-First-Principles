# Book Context: LLM Inference from First Principles

## Book

- id: `llm-inference`
- project built by reader: `rvllm`
- chapters: `ch00` to `ch20`

## What The Reader Builds

An inference engine from first principles, progressing from foundational concepts to architecture, caching, scheduling, batching, APIs, and production-oriented behavior.

## Chapter Progression

- Early chapters: inference fundamentals and architecture model.
- Middle chapters: model internals, KV cache, generation loop.
- Later chapters: paged memory, scheduling, batching, serving, and production techniques.

## Key Paths

- chapter content: `chapters/`
- chapter specs: `spec/chNN/`
- validation: `spec/chNN/validation/test_chNN.py`
- language runner docs: `spec/runners/README.md`

## Validation Notes

Validation is subprocess-driven and language-agnostic. The reader provides chapter-specific env vars such as `RVLLM_CH01_BIN`.

## Agent Guidance

- Keep guidance chapter-scoped and concrete.
- Prefer references to interface specs and expected outputs.
- Help readers map tests to their language/toolchain.
- Avoid skipping implementation details the chapter expects the reader to build.
