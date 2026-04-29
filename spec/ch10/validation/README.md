# Chapter 10 Validation

## What's tested

- All 6 section headers present
- Block concepts: block mentioned, block size 16, slot mapping, block IDs
- Slot mapping: offset and physical slot computation
- Comparison: contiguous vs paged, waste/fragmentation terms
- Benefits: zero fragmentation, dynamic growth
- Closing message

## Running

```bash
RVLLM_CH10_BIN="cargo run --example ch10_paged_attention" pytest test_ch10.py -v
```
