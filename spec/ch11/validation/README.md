# Chapter 11 Validation

## What's tested

- All 6 section headers present
- Batching concepts: static and continuous batching, iteration-level scheduling
- Mixed batches: prefill and decode mentioned
- New types: Waiting, Running, Finished (SequenceStatus states)
- Throughput comparison: utilization, idle slots, throughput metric
- Closing message

## Running

```bash
RVLLM_CH11_BIN="cargo run --example ch11_continuous_batching" pytest test_ch11.py -v
```
