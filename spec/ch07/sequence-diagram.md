# Chapter 7 -- Sequence Diagram: The Skeleton Speaks

## Diagram 1: Full Generation Flow

```mermaid
sequenceDiagram
    participant CLI as CLI (main)
    participant Tok as Tokenizer
    participant Model as Gpt2Model
    participant Cache as KV Cache
    participant Sam as GreedySampler

    Note over CLI: === SETUP ===
    CLI->>Model: Load model (openai-community/gpt2)
    CLI->>Tok: Load tokenizer
    CLI->>Sam: Create GreedySampler

    Note over CLI: === ENCODE ===
    CLI->>Tok: encode("The future of artificial intelligence is")
    Tok-->>CLI: [464, 2003, 286, 11666, 4430, 318]  (6 tokens)

    Note over CLI: === PREFILL ===
    CLI->>Model: reset_cache()
    Model->>Cache: Clear all 12 layers

    CLI->>Model: forward(token_ids=[6 tokens], pos_offset=0)
    Note over Model: Full forward pass through 12 blocks
    Model->>Cache: Populate cache: [1, 12, 6, 64] per layer
    Model-->>CLI: logits [1, 6, 50257]

    CLI->>Sam: sample(logits)
    Sam->>Sam: argmax(logits[0, 5, :])
    Sam-->>CLI: first_token (e.g., 407 = " a")

    Note over CLI: === DECODE LOOP ===
    CLI->>CLI: generated = [first_token], position = 6

    loop step 1..max_tokens
        alt next_token == EOS (50256)
            Note over CLI: STOP — EOS reached
        else
            CLI->>Model: forward([next_token], pos_offset=position)
            Model->>Cache: Append to cache: [1, 12, pos+1, 64]
            Model-->>CLI: logits [1, 1, 50257]

            CLI->>Sam: sample(logits)
            Sam-->>CLI: next_token

            CLI->>CLI: generated.append(next_token)
            CLI->>CLI: position += 1
        end
    end

    Note over CLI: === OUTPUT ===
    CLI->>Tok: decode(generated)
    Tok-->>CLI: "a key part of the ..."

    CLI->>CLI: Print prompt + text + stats
```
**Figure 7.4** — Complete generation flow: encode → prefill → decode loop → output.

## Diagram 2: KV Cache Growth During Generation

```mermaid
sequenceDiagram
    participant Step as Decode Step
    participant Cache as KV Cache (per layer)

    Note over Cache: After prefill: [1, 12, 6, 64]

    Step->>Cache: Step 1: append token at pos 6
    Note over Cache: [1, 12, 7, 64]

    Step->>Cache: Step 2: append token at pos 7
    Note over Cache: [1, 12, 8, 64]

    Step->>Cache: Step 3: append token at pos 8
    Note over Cache: [1, 12, 9, 64]

    Note over Cache: ...continues growing...

    Step->>Cache: Step N: append token at pos 6+N
    Note over Cache: [1, 12, 6+N, 64]

    Note over Step,Cache: Each step: Q attends to ALL cached K,V<br>Only new K,V computed — past reused
```
**Figure 7.5** — KV cache grows by one position per decode step. This is why generation without a cache would be O(n²) — each step would recompute all previous attention.
