# Chapter 19 -- Component Diagrams: Parallelism

## Tensor Parallelism -- Weight Splitting with AllReduce

```mermaid
flowchart TB
    subgraph "Input (same on all GPUs)"
        X["x<br/>[batch, seq, hidden_dim]"]
    end

    subgraph "Column-Parallel (QKV)"
        G0_qkv["GPU 0<br/>W_qkv[:, 0:3H/4]"]
        G1_qkv["GPU 1<br/>W_qkv[:, 3H/4:6H/4]"]
        G2_qkv["GPU 2<br/>W_qkv[:, 6H/4:9H/4]"]
        G3_qkv["GPU 3<br/>W_qkv[:, 9H/4:3H]"]
    end

    subgraph "Attention (independent per GPU)"
        G0_attn["GPU 0: Attention<br/>heads 0-7"]
        G1_attn["GPU 1: Attention<br/>heads 8-15"]
        G2_attn["GPU 2: Attention<br/>heads 16-23"]
        G3_attn["GPU 3: Attention<br/>heads 24-31"]
    end

    subgraph "Row-Parallel (Output Proj)"
        G0_out["GPU 0: W_o[0:H/4, :]"]
        G1_out["GPU 1: W_o[H/4:2H/4, :]"]
        G2_out["GPU 2: W_o[2H/4:3H/4, :]"]
        G3_out["GPU 3: W_o[3H/4:H, :]"]
    end

    AR["AllReduce<br/>Sum partial results<br/>across all 4 GPUs"]

    subgraph "Output (same on all GPUs)"
        Y["y<br/>[batch, seq, hidden_dim]"]
    end

    X --> G0_qkv & G1_qkv & G2_qkv & G3_qkv
    G0_qkv --> G0_attn
    G1_qkv --> G1_attn
    G2_qkv --> G2_attn
    G3_qkv --> G3_attn
    G0_attn --> G0_out
    G1_attn --> G1_out
    G2_attn --> G2_out
    G3_attn --> G3_out
    G0_out & G1_out & G2_out & G3_out --> AR
    AR --> Y
```
**Figure 19.1** -- Tensor parallelism for one attention layer with TP=4. Input is replicated to all GPUs. QKV projection is column-parallel (each GPU computes a slice). Output projection is row-parallel (partial results summed via AllReduce).

## Pipeline Parallelism -- Layers Distributed Across Stages

```mermaid
flowchart LR
    subgraph "Stage 0<br/>GPU 0"
        L0["Layer 0"]
        L1["Layer 1"]
        L2["..."]
        L7["Layer 7"]
        L0 --> L1 --> L2 --> L7
    end

    subgraph "Stage 1<br/>GPU 1"
        L8["Layer 8"]
        L9["Layer 9"]
        L10["..."]
        L15["Layer 15"]
        L8 --> L9 --> L10 --> L15
    end

    subgraph "Stage 2<br/>GPU 2"
        L16["Layer 16"]
        L17["Layer 17"]
        L18["..."]
        L23["Layer 23"]
        L16 --> L17 --> L18 --> L23
    end

    subgraph "Stage 3<br/>GPU 3"
        L24["Layer 24"]
        L25["Layer 25"]
        L26["..."]
        L31["Layer 31"]
        L24 --> L25 --> L26 --> L31
    end

    L7 -->|"hidden state<br/>transfer"| L8
    L15 -->|"hidden state<br/>transfer"| L16
    L23 -->|"hidden state<br/>transfer"| L24
```
**Figure 19.2** -- Pipeline parallelism with PP=4. The 32-layer model is split into 4 stages of 8 layers each. Hidden states are transferred between stages via point-to-point communication.

## Pipeline Bubble -- Idle Time Without batch_queue

```mermaid
gantt
    title Pipeline Timeline — Naive (2 Batches, 4 Stages)
    dateFormat X
    axisFormat %s

    section Stage 0
    Batch 0   :b0s0, 0, 1
    idle      :done, i0a, 1, 4
    Batch 1   :b1s0, 4, 5
    idle      :done, i0b, 5, 8

    section Stage 1
    idle      :done, i1a, 0, 1
    Batch 0   :b0s1, 1, 2
    idle      :done, i1b, 2, 5
    Batch 1   :b1s1, 5, 6
    idle      :done, i1c, 6, 8

    section Stage 2
    idle      :done, i2a, 0, 2
    Batch 0   :b0s2, 2, 3
    idle      :done, i2b, 3, 6
    Batch 1   :b1s2, 6, 7
    idle      :done, i2c, 7, 8

    section Stage 3
    idle      :done, i3a, 0, 3
    Batch 0   :b0s3, 3, 4
    idle      :done, i3b, 4, 7
    Batch 1   :b1s3, 7, 8
```
**Figure 19.3** -- Pipeline bubble with naive scheduling. Each stage is active for only 2 out of 8 timesteps (25% utilization). The grey "idle" blocks are pipeline bubbles -- wasted GPU time.

## Combined TP+PP -- 8 GPU Layout

```mermaid
flowchart TB
    subgraph "Node 0 — Stage 0 (Layers 0-15)"
        direction LR
        G0["GPU 0<br/>heads 0-7<br/>dim 0-1023"]
        G1["GPU 1<br/>heads 8-15<br/>dim 1024-2047"]
        G2["GPU 2<br/>heads 16-23<br/>dim 2048-3071"]
        G3["GPU 3<br/>heads 24-31<br/>dim 3072-4095"]
        G0 <-->|"AllReduce<br/>(NVLink)"| G1
        G1 <-->|"AllReduce<br/>(NVLink)"| G2
        G2 <-->|"AllReduce<br/>(NVLink)"| G3
    end

    subgraph "Node 1 — Stage 1 (Layers 16-31)"
        direction LR
        G4["GPU 4<br/>heads 0-7<br/>dim 0-1023"]
        G5["GPU 5<br/>heads 8-15<br/>dim 1024-2047"]
        G6["GPU 6<br/>heads 16-23<br/>dim 2048-3071"]
        G7["GPU 7<br/>heads 24-31<br/>dim 3072-4095"]
        G4 <-->|"AllReduce<br/>(NVLink)"| G5
        G5 <-->|"AllReduce<br/>(NVLink)"| G6
        G6 <-->|"AllReduce<br/>(NVLink)"| G7
    end

    G0 & G1 & G2 & G3 -->|"Hidden state<br/>(InfiniBand)"| G4 & G5 & G6 & G7
```
**Figure 19.4** -- Combined TP+PP with 8 GPUs. TP=4 within each node (high-bandwidth NVLink for AllReduce). PP=2 across nodes (InfiniBand for hidden state transfers between stages).
