# Chapter 8 -- Sequence Diagram: Fit and Finish

## Diagram 1: Generate with Timing Breakdown

```mermaid
sequenceDiagram
    participant CLI
    participant Timer
    participant Model as Gpt2Model
    participant Sam as GreedySampler
    participant Tok as Tokenizer

    CLI->>Model: Load model
    CLI->>Tok: Load tokenizer
    CLI->>Sam: Create sampler

    CLI->>Tok: encode(prompt) → 6 tokens
    CLI->>Model: reset_cache()

    Note over CLI,Timer: === PREFILL (timed) ===
    CLI->>Timer: start_prefill()
    CLI->>Model: forward(6 tokens, pos=0) → logits
    CLI->>Sam: sample(logits) → first_token
    CLI->>Timer: stop_prefill()

    Note over CLI,Timer: === DECODE (timed) ===
    CLI->>Timer: start_decode()
    loop until EOS or max_tokens
        CLI->>Model: forward(1 token, pos=N) → logits
        CLI->>Sam: sample(logits) → next_token
    end
    CLI->>Timer: stop_decode()

    CLI->>Tok: decode(all_generated) → text

    Note over CLI: === OUTPUT ===
    CLI->>CLI: Print generated text
    CLI->>Timer: prefill_time, decode_time, total_time
    CLI->>Model: cache_size_bytes() → memory
    CLI->>CLI: Print stats with timing breakdown + KV cache size
```
**Figure 8.3** — Generation flow with separate prefill and decode timing plus KV cache memory reporting.

## Diagram 2: KV Cache Memory Calculation

```mermaid
flowchart LR
    subgraph "Per Position Per Layer"
        K["K: [12 heads × 64 dim × 4 bytes]<br>= 3072 bytes"]
        V["V: [12 heads × 64 dim × 4 bytes]<br>= 3072 bytes"]
    end

    subgraph "Per Position"
        POS["12 layers × 6144 bytes<br>= 73,728 bytes<br>≈ 72 KB"]
    end

    subgraph "Total (206 positions)"
        TOTAL["206 × 72 KB<br>≈ 14.8 MB"]
    end

    K --> POS
    V --> POS
    POS --> TOTAL
```
**Figure 8.4** — KV cache memory breakdown for GPT-2 with a 206-position sequence (6 prompt + 200 generated).
