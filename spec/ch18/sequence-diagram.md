# Chapter 18 -- Sequence Diagram: Structured Output

## Diagram 1: Constrained Decode Step

```mermaid
sequenceDiagram
    participant Engine as Engine Loop
    participant PS as PipelineSampler
    participant GC as GrammarConstraint
    participant Temp as Temperature
    participant TK as TopK

    Note over Engine: Decode step with grammar constraint active

    Engine->>PS: sample(logits, tokens_so_far)

    PS->>PS: last_logits = logits[0, -1, :]

    Note over PS: Apply processor pipeline in order

    PS->>GC: process(logits, tokens_so_far)
    Note over GC: FSM state = 0<br/>Check each token against FSM<br/>Mask invalid tokens to -inf
    GC->>GC: "forty" -> advance(0, "forty") = None -> -inf
    GC->>GC: "42" -> advance(0, "42") = Some(1) -> keep
    GC-->>PS: masked logits (5 of 13 tokens survive)

    PS->>Temp: process(masked_logits, _)
    Note over Temp: Scale surviving logits<br/>by 1/0.8
    Temp-->>PS: scaled logits

    PS->>TK: process(scaled_logits, _)
    Note over TK: Keep top 3<br/>among valid tokens
    TK-->>PS: filtered logits

    Note over PS: softmax -> sample from valid tokens only
    PS-->>Engine: token_id = 2 ("42")

    Note over Engine: Advance FSM after sampling
    Engine->>GC: accept_token(2)
    Note over GC: advance(state=0, "42")<br/>state: 0 -> 1
```
**Figure 18.4** — A single constrained decode step. GrammarConstraint runs first in the pipeline, masking invalid tokens. Temperature and top-k only see valid tokens. After sampling, the engine advances the FSM state.

## Diagram 2: Multi-Step Constrained Generation

```mermaid
sequenceDiagram
    participant Engine as Engine Loop
    participant GC as GrammarConstraint
    participant Model as Model

    Note over Engine: Generate digits matching [0-9]+

    rect rgb(240, 248, 255)
        Note over GC: Step 1: FSM state = 0 (initial)
        Engine->>Model: forward(input_tokens)
        Model-->>Engine: logits
        Engine->>GC: process(logits, [])
        Note over GC: Valid: "4","2","42","0","9"<br/>Invalid: "forty","two","hello",...
        GC-->>Engine: masked logits
        Note over Engine: Sample -> Token 2 ("42")
        Engine->>GC: accept_token(2)
        Note over GC: state: 0 -> 1 (accepting)
    end

    rect rgb(240, 255, 240)
        Note over GC: Step 2: FSM state = 1 (accepting)
        Engine->>Model: forward(input_tokens + ["42"])
        Model-->>Engine: logits
        Engine->>GC: process(logits, [2])
        Note over GC: Valid: "4","2","42","0","9"<br/>(same digits valid from state 1)
        GC-->>Engine: masked logits
        Note over Engine: Sample -> Token 6 ("9")
        Engine->>GC: accept_token(6)
        Note over GC: state: 1 -> 1 (still accepting)
    end

    rect rgb(255, 248, 240)
        Note over GC: Step 3: FSM state = 1 (accepting)
        Engine->>Model: forward(input_tokens + ["42", "9"])
        Model-->>Engine: logits
        Note over Engine: FSM in accept state<br/>Can stop if EOS or max_tokens
        Note over Engine: Result: "429" — matches [0-9]+
    end
```
**Figure 18.5** — Multi-step constrained generation. Each step masks logits according to the current FSM state, samples a valid token, and advances the FSM. The FSM reaches an accepting state after the first digit, and stays accepting as more digits are generated.

## Diagram 3: Unconstrained vs Constrained Comparison

```mermaid
sequenceDiagram
    participant Engine as Engine Loop
    participant UC as Unconstrained<br/>Sampler
    participant CC as Constrained<br/>Sampler

    Note over Engine: Same logits, two approaches

    rect rgb(255, 240, 240)
        Note over UC: No grammar constraint
        Engine->>UC: sample(logits, [])
        Note over UC: All 13 tokens eligible<br/>"forty" has highest logit (5.0)
        UC-->>Engine: Token 9 ("forty")
        Note over Engine: Output: "forty" — not a number
    end

    rect rgb(240, 255, 240)
        Note over CC: With [0-9]+ constraint
        Engine->>CC: sample(logits, [])
        Note over CC: GrammarConstraint masks<br/>8 tokens to -inf<br/>"42" has highest valid logit (2.0)
        CC-->>Engine: Token 2 ("42")
        Note over Engine: Output: "42" — valid number
    end

    Note over Engine: Same model, same logits<br/>Grammar constraint guarantees structure
```
**Figure 18.6** — Unconstrained vs constrained sampling with the same logits. Without constraints, the model picks "forty" (highest logit). With `[0-9]+` constraint, non-digit tokens are masked and the model picks "42" (highest valid logit).
