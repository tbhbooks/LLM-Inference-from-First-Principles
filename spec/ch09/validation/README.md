# Chapter 9 Validation

## What's tested

- All 6 section headers present
- Internal fragmentation: waste percentage (52.2%), slot counts
- External fragmentation: allocation failure, free/largest block counts
- Scaling wall: 3 requests fit
- OS analogy: page/block terminology
- ASCII memory visualization
- Closing message

## Running

```bash
RVLLM_CH09_BIN="cargo run --example ch09_memory_problem" pytest test_ch09.py -v
```
