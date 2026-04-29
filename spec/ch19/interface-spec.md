# Chapter 19 -- Interface Specification: Parallelism

This is a language-agnostic specification. It defines the concepts, formulas,
and simulator contracts for tensor parallelism (TP) and pipeline parallelism (PP).

---

## 1. Overview

A single GPU can only hold so much model. LLaMA-7B needs ~14 GB in float16 --
that fits on one A100. LLaMA-70B needs ~140 GB -- it doesn't fit anywhere.
You need to split the model across multiple GPUs.

Two fundamental strategies:

- **Tensor Parallelism (TP)** -- Split each layer's weights across GPUs.
  Every GPU participates in every layer. Communication after every layer
  (AllReduce).
- **Pipeline Parallelism (PP)** -- Split layers across GPUs. Each GPU owns
  a subset of layers ("stage"). Hidden states flow between stages.
  Communication only at stage boundaries.

This chapter builds a simulator -- no real GPUs needed. The simulator computes
communication volumes, draws pipeline timelines, and shows how batch_queue
fills pipeline bubbles.

---

## 2. Dependencies

Conceptual understanding of transformer architecture from ch05-06:
- Attention mechanism: Q, K, V projections and output projection
- MLP: two linear layers with activation
- Layer normalization
- Hidden dimension, number of heads, number of layers

No code dependencies. This is a standalone simulator.

---

## 3. Tensor Parallelism Concepts

### 3.1 Column-Parallel Linear

Split the weight matrix along columns (the output dimension).

```
Full weight: W [hidden_dim, output_dim]
GPU 0: W_0 [hidden_dim, output_dim / num_gpus]
GPU 1: W_1 [hidden_dim, output_dim / num_gpus]
...
GPU N: W_N [hidden_dim, output_dim / num_gpus]
```

Each GPU receives the same input `x` and computes a slice of the output.
No communication needed during the forward pass of this layer -- each GPU
independently computes its portion.

**Used for:** QKV projection (column-parallel on the `3 * hidden_dim` output).

### 3.2 Row-Parallel Linear

Split the weight matrix along rows (the input dimension).

```
Full weight: W [input_dim, output_dim]
GPU 0: W_0 [input_dim / num_gpus, output_dim]
GPU 1: W_1 [input_dim / num_gpus, output_dim]
...
GPU N: W_N [input_dim / num_gpus, output_dim]
```

Each GPU computes a partial output. The partial outputs must be summed
across all GPUs -- this is the **AllReduce** operation.

**Used for:** Output projection after attention, second MLP linear layer.

### 3.3 AllReduce

AllReduce sums a tensor across all GPUs and distributes the result back
to every GPU. After AllReduce, every GPU has the identical full result.

**Ring AllReduce** is the standard algorithm:
- Data is split into `num_gpus` chunks
- Each GPU sends/receives in a ring topology
- Completes in `2 * (num_gpus - 1)` steps
- Total data transferred per GPU: `2 * tensor_size * (num_gpus - 1) / num_gpus`

### 3.4 TP Communication per Transformer Layer

Each transformer layer requires **two AllReduce** operations:
1. After the attention output projection (row-parallel)
2. After the second MLP linear layer (row-parallel)

The tensor being reduced has shape `[batch_size, seq_len, hidden_dim]`.

```
allreduce_volume_per_op = batch_size * seq_len * hidden_dim * sizeof(float)
                          * 2 * (num_gpus - 1) / num_gpus

// sizeof(float) = 4 bytes for float32, 2 bytes for float16
// Factor of 2 in ring AllReduce: reduce-scatter + all-gather phases
// (num_gpus - 1) / num_gpus: ring AllReduce efficiency factor

total_volume_per_layer = 2 * allreduce_volume_per_op    // attention + MLP
total_volume_all_layers = total_volume_per_layer * num_layers
```

**Simplified formula (used in demo):**

```
per_layer_bytes = 2 * batch_size * seq_len * hidden_dim * bytes_per_element
                  * 2 * (num_gpus - 1) / num_gpus
```

Where the first `2` is for attention + MLP, and the second `2 * (num_gpus - 1) / num_gpus`
is the ring AllReduce volume factor.

---

## 4. Pipeline Parallelism Concepts

### 4.1 Stages

The model's layers are split into contiguous groups called **stages**.
Each stage runs on a separate GPU (or group of GPUs if combined with TP).

```
Total layers: L
Number of stages: S
Layers per stage: L / S

Stage 0: layers [0, L/S)
Stage 1: layers [L/S, 2*L/S)
...
Stage S-1: layers [(S-1)*L/S, L)
```

### 4.2 Hidden State Transfer

Between stages, the hidden state tensor must be sent from one GPU to
another. This is point-to-point communication (send/recv), not AllReduce.

```
transfer_volume = batch_size * seq_len * hidden_dim * bytes_per_element
```

This is much smaller than AllReduce volume because it's a single tensor,
not a sum-and-broadcast.

### 4.3 Pipeline Bubbles

The fundamental problem with pipeline parallelism: **idle time**.

With naive sequential execution:
- Batch 0 enters stage 0 at time 0
- Batch 0 enters stage 1 at time 1
- While stage 1 processes batch 0, stage 0 is **idle** (bubble)
- Batch 0 finishes all stages before batch 1 starts

**Bubble fraction (naive):**

```
bubble_fraction = (num_stages - 1) / total_timesteps
```

For `S` stages and `B` batches processed sequentially:

```
total_timesteps = B * S + (S - 1)      // B batches × S stages + startup
// Actually simpler: each batch takes S timesteps, and they don't overlap
total_timesteps = S + (B - 1) * S = B * S
active_slots = B * S
total_slots = B * S * S               // S stages × (B * S) timesteps
utilization = active_slots / total_slots  // ... but this counts per-stage

// Simpler formulation:
// With B batches, no overlap: timeline length = B * S timesteps
// Each stage is active for B timesteps out of B * S total
// Utilization per stage = B / (B * S) = 1 / S
```

### 4.4 batch_queue Algorithm

Fill pipeline bubbles by scheduling the next micro-batch as soon as a
stage becomes free.

```
function pp_timeline_with_batch_queue(num_stages, num_batches):
    // Timeline: grid[stage][timestep] = batch_id or IDLE
    timeline_length = num_stages + num_batches - 1
    grid = array[num_stages][timeline_length] filled with IDLE

    for batch in 0..num_batches:
        for stage in 0..num_stages:
            timestep = batch + stage
            grid[stage][timestep] = batch

    return grid
```

**Utilization with batch_queue:**

```
active_slots = num_batches * num_stages
total_slots = num_stages * (num_stages + num_batches - 1)
utilization = active_slots / total_slots
bubble_fraction = 1 - utilization
              = (num_stages - 1) / (num_stages + num_batches - 1)
```

As `num_batches` grows large, utilization approaches 100%.

---

## 5. Communication Volume Formulas

### 5.1 Tensor Parallelism

| Parameter | Formula |
|-----------|---------|
| AllReduce per op | `batch * seq * hidden * elem_size * 2 * (G-1) / G` |
| AllReduce per layer | `2 * (per op)` (attention + MLP) |
| Total all layers | `per_layer * num_layers` |
| Heads per GPU | `num_heads / num_gpus` |
| Hidden per GPU | `hidden_dim / num_gpus` |

Where `G` = `num_gpus`, `elem_size` = bytes per element (4 for float32, 2 for float16).

### 5.2 Pipeline Parallelism

| Parameter | Formula |
|-----------|---------|
| Transfer per boundary | `batch * seq * hidden * elem_size` |
| Transfers per batch | `num_stages - 1` |
| Naive utilization | `1 / num_stages` (per stage) |
| batch_queue utilization | `num_batches / (num_stages + num_batches - 1)` |
| Bubble fraction (naive) | `(num_stages - 1) / num_stages` |
| Bubble fraction (batch_queue) | `(num_stages - 1) / (num_stages + num_batches - 1)` |

---

## 6. Combined TP+PP

In practice, large models use both:

- **TP within a node** -- GPUs connected by NVLink (high bandwidth, ~600 GB/s).
  AllReduce is fast.
- **PP across nodes** -- Nodes connected by InfiniBand (~400 Gb/s).
  Only hidden state transfers between stages, lower bandwidth is acceptable.

```
Total GPUs = TP_size * PP_size

Example: TP=4, PP=2 → 8 GPUs
  Node 0 (4 GPUs): Stage 0, layers 0-15, each GPU holds 1/4 of each layer
  Node 1 (4 GPUs): Stage 1, layers 16-31, each GPU holds 1/4 of each layer
```

### 6.1 GPU Layout

```
GPU layout for TP=4, PP=2, 32 layers:

  Stage 0 (layers 0-15):
    GPU 0: heads 0-7, hidden 0-1023
    GPU 1: heads 8-15, hidden 1024-2047
    GPU 2: heads 16-23, hidden 2048-3071
    GPU 3: heads 24-31, hidden 3072-4095

  Stage 1 (layers 16-31):
    GPU 4: heads 0-7, hidden 0-1023
    GPU 5: heads 8-15, hidden 1024-2047
    GPU 6: heads 16-23, hidden 2048-3071
    GPU 7: heads 24-31, hidden 3072-4095
```

---

## 7. Simulator Functions

### 7.1 TPConfig

```
struct TPConfig:
    num_gpus: int         // TP degree
    hidden_dim: int       // model hidden dimension
    num_heads: int        // total attention heads
    num_layers: int       // total transformer layers
    batch_size: int       // inference batch size
    seq_len: int          // sequence length
    bytes_per_element: int  // 4 for float32, 2 for float16
```

Validation:
- `hidden_dim % num_gpus == 0`
- `num_heads % num_gpus == 0`
- `num_gpus >= 1`

### 7.2 PPConfig

```
struct PPConfig:
    num_stages: int               // PP degree
    layers_per_stage: int         // layers assigned per stage
    batch_queue_size: int         // micro-batches for overlap (0 = naive)
    forward_time_per_stage: float // time units per stage per batch
```

Validation:
- `num_stages >= 1`
- `layers_per_stage >= 1`

### 7.3 tp_communication_volume

```
function tp_communication_volume(config: TPConfig) -> CommunicationResult:
    per_op = config.batch_size * config.seq_len * config.hidden_dim
             * config.bytes_per_element
             * 2 * (config.num_gpus - 1) / config.num_gpus
    per_layer = 2 * per_op    // attention + MLP
    total = per_layer * config.num_layers

    return {
        per_layer_bytes: per_layer,
        total_bytes: total,
        heads_per_gpu: config.num_heads / config.num_gpus,
        hidden_per_gpu: config.hidden_dim / config.num_gpus
    }
```

### 7.4 pp_pipeline_timeline

```
function pp_pipeline_timeline(config: PPConfig, num_batches: int, use_batch_queue: bool) -> Timeline:
    if use_batch_queue:
        length = config.num_stages + num_batches - 1
    else:
        length = config.num_stages * num_batches

    grid = array[config.num_stages][length] filled with IDLE

    if use_batch_queue:
        for batch in 0..num_batches:
            for stage in 0..config.num_stages:
                grid[stage][batch + stage] = batch
    else:
        for batch in 0..num_batches:
            for stage in 0..config.num_stages:
                grid[stage][batch * config.num_stages + stage] = batch

    return grid
```

### 7.5 pp_utilization

```
function pp_utilization(config: PPConfig, num_batches: int) -> UtilizationResult:
    S = config.num_stages

    // Naive
    naive_total = S * (S * num_batches)
    naive_active = S * num_batches
    naive_util = naive_active / naive_total    // = 1 / S

    // With batch_queue
    queue_total = S * (S + num_batches - 1)
    queue_active = S * num_batches
    queue_util = queue_active / queue_total

    naive_bubble = 1.0 - naive_util
    queue_bubble = 1.0 - queue_util

    return {
        naive_utilization: naive_util,
        naive_bubble_fraction: naive_bubble,
        queue_utilization: queue_util,
        queue_bubble_fraction: queue_bubble
    }
```

### 7.6 combined_config_summary

```
function combined_config_summary(tp: TPConfig, pp: PPConfig) -> Summary:
    total_gpus = tp.num_gpus * pp.num_stages

    stages = []
    for s in 0..pp.num_stages:
        first_layer = s * pp.layers_per_stage
        last_layer = (s + 1) * pp.layers_per_stage - 1
        gpu_assignments = []
        for g in 0..tp.num_gpus:
            gpu_id = s * tp.num_gpus + g
            head_start = g * (tp.num_heads / tp.num_gpus)
            head_end = (g + 1) * (tp.num_heads / tp.num_gpus) - 1
            dim_start = g * (tp.hidden_dim / tp.num_gpus)
            dim_end = (g + 1) * (tp.hidden_dim / tp.num_gpus) - 1
            gpu_assignments.append({gpu_id, head_start..head_end, dim_start..dim_end})
        stages.append({first_layer..last_layer, gpu_assignments})

    return { total_gpus, stages }
```

---

## 8. Demo Program

The demo (`examples/ch19_parallelism`) is a pure simulator. No GPU needed.

### 8.1 Part 1: Tensor Parallelism Communication

Configure a LLaMA-7B-like model:
- `hidden_dim = 4096`, `num_heads = 32`, `num_layers = 32`
- `batch_size = 1`, `seq_len = 512`, `bytes_per_element = 2` (float16)
- `num_gpus = 4`

Show:
- Heads per GPU: 32 / 4 = 8
- Hidden dim per GPU: 4096 / 4 = 1024
- AllReduce volume per layer
- Total AllReduce volume for all 32 layers
- Column-parallel and row-parallel explanation

### 8.2 Part 2: Pipeline Parallelism -- Naive

Configure:
- 4 stages, 8 layers per stage (32 total)
- 2 batches, no batch_queue

Show:
- Timeline grid: which stage processes which batch at each timestep
- Bubble/idle slots marked
- Utilization percentage
- Bubble fraction

### 8.3 Part 3: Pipeline Parallelism -- batch_queue

Same 4 stages, but 4 micro-batches with overlap.

Show:
- Overlapped timeline grid
- Fewer idle slots
- Improved utilization percentage
- Comparison with naive

### 8.4 Part 4: Combined TP+PP

Configure:
- TP=4 (within each stage), PP=2 (2 stages)
- 8 GPUs total, 32 layers, 32 heads

Show:
- Layer distribution: stage 0 = layers 0-15, stage 1 = layers 16-31
- Head distribution within each stage
- GPU layout table

### 8.5 Part 5: Scaling Guide

Table of model sizes vs GPU configurations:

| Model | Params | Weight Memory | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|-------|--------|--------------|-------|--------|--------|--------|
| 7B | 7B | ~14 GB (fp16) | TP=1 | TP=2 | TP=4 | TP=4,PP=2 |
| 13B | 13B | ~26 GB (fp16) | -- | TP=2 | TP=4 | TP=4,PP=2 |
| 70B | 70B | ~140 GB (fp16) | -- | -- | TP=4 | TP=4,PP=2 |

With estimated communication overhead for each config.

### 8.6 Output

End with: `"Chapter 19 complete. Next: Where to Go From Here (ch20)"`

---

## 9. Correctness Criteria

1. **TP splitting is valid:** `hidden_dim % num_gpus == 0` and
   `num_heads % num_gpus == 0`.
2. **AllReduce volume formula is correct:** Uses ring AllReduce formula with
   `2 * (num_gpus - 1) / num_gpus` factor.
3. **Pipeline timeline is accurate:** Each batch passes through all stages
   in order. No stage processes two batches simultaneously.
4. **Naive pipeline has correct bubble fraction:** `(S - 1) / S` per stage.
5. **batch_queue improves utilization:** Utilization with batch_queue >
   utilization without, for the same number of batches.
6. **Combined TP+PP:** Total GPUs = `TP_size * PP_size`. Layers distributed
   evenly across stages. Heads distributed evenly within each stage.
7. **Scaling guide is reasonable:** Larger models require more GPUs.
   TP preferred within node, PP across nodes.

---

## 10. Validation Summary

| Test | What it checks |
|------|---------------|
| Part 1 present | Tensor parallelism section exists |
| Part 2 present | Naive pipeline parallelism section exists |
| Part 3 present | batch_queue pipeline parallelism section exists |
| Part 4 present | Combined TP+PP section exists |
| Part 5 present | Scaling guide section exists |
| TP concepts | AllReduce mentioned, communication volume shown |
| PP concepts | Pipeline/bubble/utilization mentioned |
| batch_queue | Overlap or batch_queue mentioned, improved utilization shown |
| Combined | TP+PP shown together, GPU count = TP * PP |
| Completion marker | "Chapter 19 complete" appears |
