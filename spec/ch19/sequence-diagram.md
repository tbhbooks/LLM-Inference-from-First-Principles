# Chapter 19 -- Sequence Diagrams: Parallelism

## TP Forward Pass -- Split, Compute, AllReduce

```mermaid
sequenceDiagram
    participant Input as Input Tensor
    participant G0 as GPU 0
    participant G1 as GPU 1
    participant G2 as GPU 2
    participant G3 as GPU 3
    participant AR as AllReduce

    Note over Input: x [batch, seq, 4096]

    Input->>G0: Broadcast x (same input to all)
    Input->>G1: Broadcast x
    Input->>G2: Broadcast x
    Input->>G3: Broadcast x

    Note over G0,G3: Column-parallel QKV projection

    par Parallel compute
        G0->>G0: W_qkv[:, 0:3072] @ x<br/>→ Q,K,V for heads 0-7
        G1->>G1: W_qkv[:, 3072:6144] @ x<br/>→ Q,K,V for heads 8-15
        G2->>G2: W_qkv[:, 6144:9216] @ x<br/>→ Q,K,V for heads 16-23
        G3->>G3: W_qkv[:, 9216:12288] @ x<br/>→ Q,K,V for heads 24-31
    end

    Note over G0,G3: Attention (independent per GPU)

    par Parallel attention
        G0->>G0: Attention(Q,K,V) → partial out
        G1->>G1: Attention(Q,K,V) → partial out
        G2->>G2: Attention(Q,K,V) → partial out
        G3->>G3: Attention(Q,K,V) → partial out
    end

    Note over G0,G3: Row-parallel output projection

    par Parallel matmul
        G0->>G0: W_o[0:1024, :] @ attn_out → partial
        G1->>G1: W_o[1024:2048, :] @ attn_out → partial
        G2->>G2: W_o[2048:3072, :] @ attn_out → partial
        G3->>G3: W_o[3072:4096, :] @ attn_out → partial
    end

    G0->>AR: partial result
    G1->>AR: partial result
    G2->>AR: partial result
    G3->>AR: partial result

    Note over AR: Sum all partial results<br/>(ring AllReduce)

    AR->>G0: full result [batch, seq, 4096]
    AR->>G1: full result [batch, seq, 4096]
    AR->>G2: full result [batch, seq, 4096]
    AR->>G3: full result [batch, seq, 4096]

    Note over G0,G3: All GPUs now have identical output.<br/>Repeat for MLP (another AllReduce).
```
**Figure 19.5** -- Tensor parallelism forward pass for one attention layer. Input is broadcast to all GPUs. Each GPU computes its slice (column-parallel QKV, independent attention, row-parallel output). AllReduce sums the partial results so every GPU has the full output.

## PP Without batch_queue -- Sequential Batches

```mermaid
sequenceDiagram
    participant S0 as Stage 0<br/>(Layers 0-7)
    participant S1 as Stage 1<br/>(Layers 8-15)
    participant S2 as Stage 2<br/>(Layers 16-23)
    participant S3 as Stage 3<br/>(Layers 24-31)

    Note over S0,S3: Batch 0 — sequential through all stages

    rect rgb(230, 245, 255)
        S0->>S0: Process Batch 0
        Note over S1,S3: IDLE (bubble)
        S0->>S1: hidden state
        S1->>S1: Process Batch 0
        Note over S0: IDLE
        Note over S2,S3: IDLE (bubble)
        S1->>S2: hidden state
        S2->>S2: Process Batch 0
        Note over S0,S1: IDLE
        Note over S3: IDLE (bubble)
        S2->>S3: hidden state
        S3->>S3: Process Batch 0
        Note over S0,S2: IDLE
    end

    Note over S0,S3: Batch 1 — only starts after Batch 0 finishes

    rect rgb(255, 243, 224)
        S0->>S0: Process Batch 1
        Note over S1,S3: IDLE (bubble)
        S0->>S1: hidden state
        S1->>S1: Process Batch 1
        Note over S0: IDLE
        Note over S2,S3: IDLE (bubble)
        S1->>S2: hidden state
        S2->>S2: Process Batch 1
        Note over S0,S1: IDLE
        Note over S3: IDLE (bubble)
        S2->>S3: hidden state
        S3->>S3: Process Batch 1
        Note over S0,S2: IDLE
    end

    Note over S0,S3: Utilization: 25%<br/>Each stage active 2/8 timesteps
```
**Figure 19.6** -- Pipeline parallelism without batch_queue. Batch 0 flows through all 4 stages before Batch 1 starts. At any given timestep, 3 out of 4 stages are idle. This is the pipeline bubble problem.

## PP With batch_queue -- Overlapped Micro-Batches

```mermaid
sequenceDiagram
    participant S0 as Stage 0<br/>(Layers 0-7)
    participant S1 as Stage 1<br/>(Layers 8-15)
    participant S2 as Stage 2<br/>(Layers 16-23)
    participant S3 as Stage 3<br/>(Layers 24-31)

    Note over S0,S3: Timestep 0 — pipeline filling

    S0->>S0: Batch 0

    Note over S0,S3: Timestep 1 — two stages active

    par
        S0->>S0: Batch 1
        S0->>S1: hidden state (B0)
        S1->>S1: Batch 0
    end

    Note over S0,S3: Timestep 2 — three stages active

    par
        S0->>S0: Batch 2
        S1->>S1: Batch 1
        S1->>S2: hidden state (B0)
        S2->>S2: Batch 0
    end

    Note over S0,S3: Timestep 3 — all stages active (pipeline full)

    par
        S0->>S0: Batch 3
        S1->>S1: Batch 2
        S2->>S2: Batch 1
        S2->>S3: hidden state (B0)
        S3->>S3: Batch 0
    end

    Note over S0,S3: Timesteps 4-6 — pipeline draining

    par
        S1->>S1: Batch 3
        S2->>S2: Batch 2
        S3->>S3: Batch 1
    end

    par
        S2->>S2: Batch 3
        S3->>S3: Batch 2
    end

    S3->>S3: Batch 3

    Note over S0,S3: Utilization: 57.1%<br/>16 active slots / 28 total<br/>Much better than 25% naive!
```
**Figure 19.7** -- Pipeline parallelism with batch_queue. Micro-batches overlap: as soon as Stage 0 finishes Batch 0, it starts Batch 1 while Stage 1 processes Batch 0. After the pipeline fills (timestep 3), all stages are active simultaneously. Bubbles only exist during pipeline fill and drain.
