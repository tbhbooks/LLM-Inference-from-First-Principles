# Chapter 12 Validation

## What's tested

- All 6 section headers present
- Three queues: waiting, running, swapped
- Scheduling: schedule() method, FCFS ordering, batch limits
- Preemption: preemption mentioned, preempted IDs shown
- Memory awareness: blocks, allocation checks, can_allocate
- SchedulerOutput: new_requests/prefill, running/decode, token counts
- Multi-step simulation with queue state changes
- Closing message

## Running

```bash
RVLLM_CH12_BIN="cargo run --example ch12_the_scheduler" pytest test_ch12.py -v
```
