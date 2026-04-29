# Chapter 19 -- LLM Prompt Template: Parallelism

Copy and paste the prompt below into your LLM of choice, or use it as a
reference while implementing by hand. This builds on conceptual understanding
from Chapters 1-18.

---

## The Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 19.
I have an existing project from Chapters 1-18 with:
- Full GPT-2 model with KV cache (ch04-06)
- PagedAttention memory management (ch09-10)
- Continuous batching scheduler (ch11-12)
- Engine loop with sampling pipeline (ch13-14)
- API server with streaming (ch15-16)
- Prefix caching (ch17)
- Speculative decoding (ch18)

Now build a parallelism simulator. This chapter is conceptual — no real
multi-GPU code. The simulator demonstrates tensor parallelism (TP) and
pipeline parallelism (PP) with computed communication volumes and pipeline
timelines.

LANGUAGE/FRAMEWORK: [Rust with candle / Python with PyTorch / your choice]

=== WHAT TO CREATE / MODIFY ===

  NEW FILES:
    examples/ch19_parallelism   <-- Simulator demo (standalone, no GPU needed)

  MODIFY:
    nothing

  KEEP UNCHANGED:
    everything — this is a standalone simulator

=== TENSOR PARALLELISM CONFIG ===

  struct TPConfig:
      num_gpus: int         // number of GPUs in the TP group
      hidden_dim: int       // model hidden dimension (e.g., 4096)
      num_heads: int        // total number of attention heads
      num_layers: int       // total number of transformer layers
      batch_size: int       // batch size
      seq_len: int          // sequence length

  Constraints:
      hidden_dim % num_gpus == 0
      num_heads % num_gpus == 0

=== PIPELINE PARALLELISM CONFIG ===

  struct PPConfig:
      num_stages: int           // number of pipeline stages (= number of GPU groups)
      layers_per_stage: int     // layers assigned to each stage
      batch_queue_size: int     // number of micro-batches to overlap (0 = naive)
      forward_time_per_stage: float  // time units for one stage to process one batch

=== TENSOR PARALLELISM CONCEPTS ===

  Column-parallel: Split weight matrix along columns.
    - QKV projection: W_qkv [hidden_dim, 3*hidden_dim] →
      each GPU gets W_qkv [hidden_dim, 3*hidden_dim / num_gpus]
    - Each GPU computes its slice of Q, K, V independently.

  Row-parallel: Split weight matrix along rows.
    - Output projection: W_o [hidden_dim, hidden_dim] →
      each GPU gets W_o [hidden_dim / num_gpus, hidden_dim]
    - Partial results summed via AllReduce.

  AllReduce communication volume per layer:
    volume_bytes = 2 * batch_size * seq_len * hidden_dim * sizeof(float) * (num_gpus - 1) / num_gpus
    // Factor of 2: one AllReduce after attention output, one after MLP output.
    // The (num_gpus - 1) / num_gpus factor is from the ring AllReduce algorithm.

=== PIPELINE PARALLELISM CONCEPTS ===

  Layers distributed across stages:
    Stage 0: layers 0..(layers_per_stage - 1)
    Stage 1: layers layers_per_stage..(2 * layers_per_stage - 1)
    ...

  Naive (no batch_queue):
    - Batch enters stage 0, completes, moves to stage 1, etc.
    - Only 1 stage active at a time → massive idle time ("pipeline bubble").
    - Bubble fraction = (num_stages - 1) / num_stages

  batch_queue (micro-batch overlap):
    - Split input into micro-batches.
    - As soon as stage 0 finishes micro-batch 0, it starts micro-batch 1.
    - Meanwhile stage 1 processes micro-batch 0.
    - Stages stay busy after the pipeline fills.
    - Bubble fraction = (num_stages - 1) / (num_stages + num_batches - 1)

=== SIMULATOR FUNCTIONS ===

  tp_communication_volume(config: TPConfig) -> CommunicationResult:
      // Computes AllReduce volume per layer and total
      per_layer_bytes = 2 * config.batch_size * config.seq_len * config.hidden_dim
                        * 4 * (config.num_gpus - 1) / config.num_gpus
      total_bytes = per_layer_bytes * config.num_layers
      return { per_layer_bytes, total_bytes }

  pp_pipeline_timeline(config: PPConfig, num_batches: int) -> Timeline:
      // Returns a grid: timeline[stage][timestep] = batch_id or IDLE
      // Without batch_queue: sequential, lots of IDLE
      // With batch_queue: overlapped, fewer IDLE slots

  pp_utilization(config: PPConfig, num_batches: int) -> UtilizationResult:
      // Naive: active_slots / total_slots
      // With batch_queue: better ratio
      naive_util = num_batches / (num_batches + num_stages - 1)
      queue_util = (num_batches * num_stages) / (num_stages * (num_stages + num_batches - 1))
      // Simplified: both approach 100% as num_batches → infinity

  combined_config_summary(tp: TPConfig, pp: PPConfig) -> Summary:
      // Total GPUs = tp.num_gpus * pp.num_stages
      // Show which layers go to which stage, how heads split within stage

=== DEMO PROGRAM (examples/ch19_parallelism) ===

The demo is a pure simulator — no GPU, no model loading. It should:

1. PART 1: Tensor Parallelism Communication
   - Configure LLaMA-7B-like model: hidden=4096, heads=32, layers=32
   - TP=4 GPUs
   - Show weight splitting (heads per GPU, hidden dim per GPU)
   - Compute AllReduce volume per layer and total
   - Print communication table

2. PART 2: Pipeline Parallelism — Naive (No batch_queue)
   - 4 stages, 2 batches, no overlap
   - Print timeline grid showing which stage is active at each timestep
   - Show bubble percentage and utilization

3. PART 3: Pipeline Parallelism — With batch_queue
   - Same 4 stages, but now 4 micro-batches with overlap
   - Print overlapped timeline grid
   - Show improved utilization
   - Compare with naive

4. PART 4: Combined TP+PP
   - TP=4 within each stage, PP=2 stages → 8 GPUs total
   - Show layer distribution across stages
   - Show head/dim distribution within each stage
   - Print GPU layout table

5. PART 5: Scaling Guide
   - Table of model sizes (7B, 13B, 70B) × GPU configs (1, 2, 4, 8 GPUs)
   - Recommended TP/PP split for each combination
   - Communication volume estimates

End with: "Chapter 19 complete. Next: Where to Go From Here (ch20)"

=== WHAT SUCCESS LOOKS LIKE ===

Running the demo produces clearly labeled output showing:
- How tensor parallelism splits weights and requires AllReduce communication
- How pipeline parallelism creates bubbles without overlapping
- How batch_queue fills pipeline bubbles by scheduling micro-batches
- How TP+PP combine for large-scale deployment
- A practical guide for choosing parallelism strategy by model size

No GPU needed. All numbers are computed from formulas.
The output demonstrates the tradeoffs of each parallelism strategy.

=== WHAT TO PRODUCE ===

Produce the complete set of NEW source files.
Do NOT recreate files from earlier chapters that are unchanged.

After this chapter, the reader understands how LLM inference engines
distribute work across multiple GPUs using tensor and pipeline parallelism.
```
