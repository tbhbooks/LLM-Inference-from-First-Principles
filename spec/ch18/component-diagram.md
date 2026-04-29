# Chapter 18 -- Component Diagram: Structured Output

## Structured Output Class Structure

```mermaid
classDiagram
    direction TB

    class LogitsProcessor {
        <<trait>>
        +process(logits, token_ids_so_far) FloatArray
    }

    class GrammarConstraint {
        fsm: SimpleFSM
        current_state: FSMState
        vocab: list~TokenId, String~
        +process(logits, token_ids_so_far) FloatArray
        +accept_token(token_id)
        +reset()
    }
    note for GrammarConstraint "Masks invalid tokens<br/>to -infinity per FSM state"

    class SimpleFSM {
        states: list~FSMState~
        initial_state: FSMState
        accept_states: set~FSMState~
        transitions: map~FSMState char, FSMState~
        +advance(state, token_str) Option~FSMState~
        +is_accepting(state) bool
    }
    note for SimpleFSM "Compiled from regex pattern"

    class StructuredOutputConfig {
        pattern: String
    }
    note for StructuredOutputConfig "User-facing config"

    class PipelineSampler {
        processors: list~LogitsProcessor~
        temperature: float
        rng: RandomGenerator
        +sample(logits, token_ids_so_far) TokenId
    }

    class TemperatureProcessor {
        temperature: float
        +process(logits, _) FloatArray
    }

    class TopKProcessor {
        k: int
        +process(logits, _) FloatArray
    }

    class TopPProcessor {
        p: float
        +process(logits, _) FloatArray
    }

    class Sampler {
        <<trait>>
        +sample(logits, token_ids_so_far) TokenId
    }

    %% Trait implementations
    GrammarConstraint ..|> LogitsProcessor : implements
    TemperatureProcessor ..|> LogitsProcessor : implements
    TopKProcessor ..|> LogitsProcessor : implements
    TopPProcessor ..|> LogitsProcessor : implements

    PipelineSampler ..|> Sampler : implements

    %% Composition
    GrammarConstraint o-- SimpleFSM : contains
    StructuredOutputConfig ..> SimpleFSM : compile_regex()
    PipelineSampler o-- LogitsProcessor : contains 0..*
```
**Figure 18.1** — Structured output class structure. GrammarConstraint implements the LogitsProcessor trait from ch14, wrapping a SimpleFSM compiled from a regex pattern. It plugs into PipelineSampler alongside temperature, top-k, and top-p processors.

## FSM State Diagram for `[0-9]+`

```mermaid
stateDiagram-v2
    direction LR

    [*] --> S0

    S0 : State 0<br/>Initial<br/>(not accepting)
    S1 : State 1<br/>Accepting

    S0 --> S1 : '0'-'9'
    S1 --> S1 : '0'-'9'
```
**Figure 18.2** — FSM state diagram for the pattern `[0-9]+`. State 0 is the initial state (not accepting). Any digit transitions to State 1 (accepting). State 1 loops on digits, allowing one or more digits total.

## Token Masking Flow

```mermaid
flowchart LR
    subgraph "Input"
        A["Raw logits<br/>[vocab_size]"]
        B["FSM state<br/>(current)"]
        C["Vocabulary<br/>(token_id, string)"]
    end

    subgraph "GrammarConstraint.process()"
        D["For each token:<br/>fsm.advance(state, token_str)"]
        E{"Valid<br/>transition?"}
        F["Keep logit"]
        G["Set logit = -inf"]
    end

    subgraph "Result"
        H["Masked logits<br/>(only valid tokens<br/>have finite values)"]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    E -->|yes| F
    E -->|no| G
    F --> H
    G --> H

    subgraph "Then: rest of pipeline"
        I["Temperature"]
        J["TopK"]
        K["TopP"]
        L["Sample"]
    end

    H --> I --> J --> K --> L
```
**Figure 18.3** — Token masking flow. For each token in the vocabulary, the GrammarConstraint checks if the FSM can consume that token's string from the current state. Invalid tokens get their logits set to negative infinity. The masked logits then flow through the rest of the sampling pipeline.
