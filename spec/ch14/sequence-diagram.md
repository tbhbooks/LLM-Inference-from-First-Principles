# Chapter 14 -- Sequence Diagram: Sampling Strategies

## Diagram 1: Pipeline Sampling Flow

```mermaid
sequenceDiagram
    participant Engine as Engine Loop
    participant Sam as PipelineSampler
    participant RP as RepetitionPenalty
    participant Temp as Temperature
    participant TK as TopK
    participant TP as TopP

    Note over Engine: Decode step — model produced logits [1, 1, 50257]

    Engine->>Sam: sample(logits, tokens_so_far=[464, 2003, 407, 257])

    Sam->>Sam: last_logits = logits[0, -1, :]

    Note over Sam: Apply processor pipeline in order

    Sam->>RP: process(logits, [464, 2003, 407, 257])
    Note over RP: Divide logits of tokens<br/>464, 2003, 407, 257<br/>by penalty (1.2)
    RP-->>Sam: modified logits

    Sam->>Temp: process(logits, _)
    Note over Temp: logits / 0.8
    Temp-->>Sam: scaled logits

    Sam->>TK: process(logits, _)
    Note over TK: Keep top 50,<br/>rest → -inf
    TK-->>Sam: filtered logits

    Sam->>TP: process(logits, _)
    Note over TP: Softmax → cumsum<br/>cutoff at p=0.9
    TP-->>Sam: nucleus-filtered logits

    Note over Sam: softmax → multinomial sample
    Sam-->>Engine: token_id = 318
```
**Figure 14.4** — A single sampling step through the full pipeline. Each processor transforms the logits in order: repetition penalty, temperature, top-k, top-p. Then softmax and multinomial selection.

## Diagram 2: SamplingParams → Pipeline Construction

```mermaid
sequenceDiagram
    participant User as User / API
    participant SP as SamplingParams
    participant PS as PipelineSampler

    User->>SP: Create SamplingParams(<br/>temperature=0.8,<br/>top_k=50,<br/>top_p=0.9,<br/>repetition_penalty=1.2)

    SP->>SP: Validate parameters

    Note over SP: Build processor list

    SP->>PS: new PipelineSampler(processors=[<br/>  RepetitionPenalty(1.2),<br/>  Temperature(0.8),<br/>  TopK(50),<br/>  TopP(0.9)<br/>], temperature=0.8)

    Note over PS: Ready for sample() calls
```
**Figure 14.5** — SamplingParams validates user input and constructs the pipeline sampler with the correct processors in order.

## Diagram 3: Greedy vs. Pipeline Sampling

```mermaid
sequenceDiagram
    participant Engine as Engine Loop
    participant GS as GreedySampler
    participant PS as PipelineSampler

    Note over Engine: Same logits, two samplers

    rect rgb(230, 245, 255)
        Note over GS: temperature = 0.0 (greedy)
        Engine->>GS: sample(logits, tokens_so_far)
        GS->>GS: argmax(logits[0, -1, :])
        GS-->>Engine: token = 407 (always the same)
    end

    rect rgb(255, 243, 224)
        Note over PS: temperature = 0.8, top_p = 0.9
        Engine->>PS: sample(logits, tokens_so_far)
        PS->>PS: apply pipeline → softmax → sample
        PS-->>Engine: token = 523 (varies each call)
    end

    Note over Engine: Greedy: deterministic, repetitive<br/>Pipeline: stochastic, diverse
```
**Figure 14.6** — Greedy always returns the same token. Pipeline sampling introduces controlled randomness for more diverse output.
