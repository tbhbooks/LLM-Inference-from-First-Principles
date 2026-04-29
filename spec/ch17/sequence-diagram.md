# Chapter 17 -- Sequence Diagram: Speculative Decoding

## Diagram 1: Full Accept Scenario

```mermaid
sequenceDiagram
    participant Engine as Engine Loop
    participant SD as SpeculativeDecoder
    participant Draft as Draft Model
    participant Target as Target Model

    Note over Engine: Generate next tokens speculatively (K=4)

    Engine->>SD: speculative_step(context=[t1, t2, t3])

    Note over SD: Draft Phase — K cheap forward passes

    SD->>Draft: forward([t1, t2, t3])
    Draft-->>SD: logits → argmax → d0
    SD->>Draft: forward([t1, t2, t3, d0])
    Draft-->>SD: logits → argmax → d1
    SD->>Draft: forward([t1, t2, t3, d0, d1])
    Draft-->>SD: logits → argmax → d2
    SD->>Draft: forward([t1, t2, t3, d0, d1, d2])
    Draft-->>SD: logits → argmax → d3

    Note over SD: DraftResult: [d0, d1, d2, d3]

    Note over SD: Verify Phase — 1 expensive forward pass

    SD->>Target: forward([t1, t2, t3, d0, d1, d2, d3])
    Target-->>SD: target_logits [L0, L1, L2, L3, L4]

    Note over SD: Greedy verification
    Note over SD: argmax(L0)=d0 ✓<br/>argmax(L1)=d1 ✓<br/>argmax(L2)=d2 ✓<br/>argmax(L3)=d3 ✓<br/>All accepted!<br/>bonus=argmax(L4)

    SD-->>Engine: VerifyResult(accepted=[d0,d1,d2,d3],<br/>bonus=d4, total=5)

    Note over Engine: Append 5 tokens to context
```
**Figure 17.4** — Full acceptance scenario. The draft model proposes 4 tokens, the target model confirms all 4 match, and a bonus 5th token is produced. Total: 5 tokens from 1 target forward pass.

## Diagram 2: Partial Accept Scenario

```mermaid
sequenceDiagram
    participant Engine as Engine Loop
    participant SD as SpeculativeDecoder
    participant Draft as Draft Model
    participant Target as Target Model

    Note over Engine: Speculative step (K=4)

    Engine->>SD: speculative_step(context)

    Note over SD: Draft Phase

    SD->>Draft: forward x4 (cheap)
    Draft-->>SD: DraftResult: [d0, d1, d2, d3]

    Note over SD: Verify Phase

    SD->>Target: forward(context + [d0, d1, d2, d3])
    Target-->>SD: target_logits [L0, L1, L2, L3, L4]

    Note over SD: Greedy verification
    Note over SD: argmax(L0)=d0 ✓ accept<br/>argmax(L1)=d1 ✓ accept<br/>argmax(L2)=t2 ✗ REJECT<br/>(t2 != d2)<br/>STOP — d3 never checked

    SD-->>Engine: VerifyResult(accepted=[d0,d1],<br/>bonus=t2, total=3)

    Note over Engine: Append 3 tokens to context<br/>Tokens d2, d3 discarded
```
**Figure 17.5** — Partial acceptance. Draft proposed 4 tokens, but the target model disagrees at position 2. Tokens d0 and d1 are accepted; the target's choice (t2) becomes the bonus. Tokens d2 and d3 are discarded. Total: 3 tokens.

## Diagram 3: Multi-Step Loop with Varying Acceptance

```mermaid
sequenceDiagram
    participant Engine as Engine Loop
    participant SD as SpeculativeDecoder
    participant Metrics as AcceptanceMetrics

    Note over Engine: Step 1 (K=4)
    Engine->>SD: speculative_step(context)
    SD-->>Engine: accepted=4, bonus=1, total=5
    Engine->>Metrics: update(drafted=4, accepted=4)
    Note over Metrics: rate=4/4=100%

    Note over Engine: Step 2 (K=4)
    Engine->>SD: speculative_step(context)
    SD-->>Engine: accepted=2, bonus=1, total=3
    Engine->>Metrics: update(drafted=4, accepted=2)
    Note over Metrics: rate=6/8=75%

    Note over Engine: Step 3 (K=4)
    Engine->>SD: speculative_step(context)
    SD-->>Engine: accepted=0, bonus=1, total=1
    Engine->>Metrics: update(drafted=4, accepted=0)
    Note over Metrics: rate=6/12=50%

    Note over Engine: After 3 steps
    Engine->>Metrics: summary()
    Metrics-->>Engine: drafted=12, accepted=6<br/>rate=50%, avg=3.0 tokens/step
```
**Figure 17.6** — Three speculative decode steps with varying acceptance. Step 1: full accept (5 tokens). Step 2: partial accept (3 tokens). Step 3: immediate reject (1 token). The running acceptance rate drops from 100% to 50%. Average tokens per target pass: 3.0 (still 3x better than standard decode).
