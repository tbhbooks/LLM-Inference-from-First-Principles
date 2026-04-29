# Chapter 17 -- Component Diagram: Speculative Decoding

## SpeculativeDecoder Structure

```mermaid
classDiagram
    direction TB

    class SpeculativeConfig {
        draft_tokens: int
        +validate() Result
    }
    note for SpeculativeConfig "K = number of tokens<br/>to draft per step"

    class DraftResult {
        proposed_tokens: list~TokenId~
        proposed_logits: list~FloatArray~
    }
    note for DraftResult "K tokens proposed<br/>by the draft model"

    class VerifyResult {
        accepted_tokens: list~TokenId~
        bonus_token: Option~TokenId~
        num_accepted: int
        +total_tokens() int
    }
    note for VerifyResult "0..K accepted + bonus<br/>= 1..K+1 total tokens"

    class SpeculativeDecoder {
        config: SpeculativeConfig
        total_drafted: int
        total_accepted: int
        +draft(context, K) DraftResult
        +verify(draft_result, target_logits) VerifyResult
        +acceptance_rate() float
        +average_tokens_per_step() float
    }

    class DraftModel {
        <<trait>>
        +forward(context) FloatArray
    }
    note for DraftModel "Small/cheap model<br/>or n-gram proposer"

    class TargetModel {
        <<trait>>
        +forward(context_with_drafts) list~FloatArray~
    }
    note for TargetModel "Full-size model<br/>verifies in 1 pass"

    class AcceptanceMetrics {
        total_drafted: int
        total_accepted: int
        +rate() float
        +avg_tokens_per_step() float
    }

    %% Relationships
    SpeculativeDecoder --> SpeculativeConfig : configured by
    SpeculativeDecoder --> DraftModel : uses for drafting
    SpeculativeDecoder --> TargetModel : uses for verification
    SpeculativeDecoder ..> DraftResult : produces
    SpeculativeDecoder ..> VerifyResult : produces
    SpeculativeDecoder --> AcceptanceMetrics : tracks
```
**Figure 17.1** — SpeculativeDecoder orchestrates the draft-then-verify loop. A cheap DraftModel proposes tokens; the full TargetModel verifies them in one pass. AcceptanceMetrics tracks how often the draft model guesses correctly.

## Draft-Then-Verify Flow

```mermaid
flowchart LR
    subgraph "Draft Phase (K cheap passes)"
        A["Context<br/>[t1, t2, ..., tn]"] --> B["Draft Model<br/>forward x K"]
        B --> C["DraftResult<br/>[d0, d1, d2, d3]"]
    end

    subgraph "Verify Phase (1 expensive pass)"
        C --> D["Target Model<br/>forward([context, d0..d3])"]
        D --> E["Target Logits<br/>[L0, L1, L2, L3, L4]"]
    end

    subgraph "Accept/Reject"
        E --> F{"argmax(Li)<br/>== di ?"}
        F -->|all match| G["Accept all K<br/>+ bonus token<br/>= K+1 tokens"]
        F -->|mismatch at i| H["Accept 0..i-1<br/>+ bonus token<br/>= i+1 tokens"]
    end

    G --> I["Append to context"]
    H --> I
```
**Figure 17.2** — The speculative decoding flow. The draft model runs K cheap forward passes to propose tokens. The target model verifies all K in one expensive forward pass. Tokens are accepted until the first mismatch; the target's choice at that position becomes the bonus token.

## Speedup Comparison

```mermaid
flowchart TB
    subgraph "Standard Decode (5 tokens)"
        S1["Pass 1<br/>1 token"] --> S2["Pass 2<br/>1 token"]
        S2 --> S3["Pass 3<br/>1 token"]
        S3 --> S4["Pass 4<br/>1 token"]
        S4 --> S5["Pass 5<br/>1 token"]
    end

    subgraph "Speculative Decode (5 tokens, K=4, full accept)"
        D1["Draft x4<br/>(cheap)"] --> V1["Verify x1<br/>(expensive)"]
        V1 --> R1["5 tokens<br/>accepted"]
    end

    subgraph "Speculative Decode (5 tokens, K=4, partial)"
        D2["Draft x4<br/>(cheap)"] --> V2["Verify x1<br/>3 tokens"]
        V2 --> D3["Draft x4<br/>(cheap)"]
        D3 --> V3["Verify x1<br/>2 tokens"]
    end
```
**Figure 17.3** — Speedup comparison. Standard decode needs 5 target forward passes for 5 tokens. With speculation and full acceptance, 1 draft+verify cycle produces all 5. With partial acceptance, 2 cycles suffice. The draft passes are cheap (small model), so the wall-clock savings can be substantial.
