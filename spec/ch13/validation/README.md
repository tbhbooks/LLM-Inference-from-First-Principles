# Chapter 13 Validation

## What's tested

- All 6 section headers present
- Engine concepts: step, schedule, forward, sample, update phases
- Request lifecycle: Waiting, Running, Finished status transitions
- Block tracking: allocation and freeing across steps
- Batching: multiple requests in the same step
- Input preparation: prefill vs decode distinction, positions shown
- Closing message

## Running

```bash
RVLLM_CH13_BIN="cargo run --example ch13_the_engine_loop" pytest test_ch13.py -v
```
