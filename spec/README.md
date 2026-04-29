# rvllm Spec Artifacts

Spec-driven development artifacts for "Building vLLM from Scratch." Each chapter directory contains everything a developer (or LLM) needs to implement that chapter's code — in any language.

## Structure

Each `chNN/` directory contains:

| File | Purpose |
|------|---------|
| `component-diagram.md` | Mermaid diagrams — structs, modules, relationships |
| `sequence-diagram.md` | Mermaid diagrams — data flow for key operations |
| `interface-spec.md` | Type signatures, method contracts, constraints |
| `expected-output.txt` | Exact stdout the implementation should produce |
| `prompt-template.md` | Ready-to-paste LLM prompt referencing the spec |
| `validation/` | Tests (Python/pytest) that verify any correct implementation |

## How to Use

### As a reader

1. Read the chapter in `../book2/`
2. Review the spec in `spec/chNN/`
3. Implement in your language of choice (or paste `prompt-template.md` into an LLM)
4. Validate: `cd spec/chNN/validation && pytest`

### As an LLM assistant

1. Read `interface-spec.md` for the contract
2. Read `expected-output.txt` for what correct looks like
3. Generate code that satisfies both
4. Run validation tests to confirm

## Running Tests

```bash
# All chapters
cd spec && pytest

# Single chapter
cd spec/ch01/validation && pytest

# Verbose
pytest -v spec/ch01/validation/
```

Requirements: Python 3.10+, pytest (`pip install pytest`)

## Reference Implementation

A Rust reference implementation exists in `../rvllm/`. Git tags (`ch01`, `ch02`, ...) snapshot the code at each chapter. This is one possible implementation — the specs are the source of truth, not the Rust code.
