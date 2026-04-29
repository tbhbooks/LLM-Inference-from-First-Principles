# Chapter 15 -- Component Diagram: Building the API Server

## API Server Structure

```mermaid
classDiagram
    direction TB

    class ApiServer {
        address: string
        model_name: string
        request_tx: Sender~InferenceRequest~
        +start() Result
        +router() Router
    }

    class CompletionRequest {
        model: string
        prompt: string
        max_tokens: int
        temperature: float
        top_k: int
        top_p: float
        repetition_penalty: float
        stream: bool
        +validate() Result
    }

    class CompletionResponse {
        id: string
        object: string
        created: int
        model: string
        choices: list~Choice~
        usage: Usage
    }

    class CompletionChunk {
        id: string
        object: string
        created: int
        choices: list~ChunkChoice~
    }

    class Choice {
        text: string
        index: int
        finish_reason: string
    }

    class ChunkChoice {
        text: string
        index: int
        finish_reason: Option~string~
    }

    class Usage {
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int
    }

    class ErrorResponse {
        error: ErrorBody
    }

    class ErrorBody {
        message: string
        type: string
        code: Option~string~
    }

    class InferenceRequest {
        request_id: string
        prompt: string
        sampling_params: SamplingParams
        max_tokens: int
        response_tx: Sender~InferenceResponse~
    }

    class InferenceResponse {
        <<enumeration>>
        Token(text, finish_reason)
        Error(message)
        Done
    }

    %% Composition
    CompletionResponse *-- Choice : contains 1..*
    CompletionResponse *-- Usage : contains 1
    CompletionChunk *-- ChunkChoice : contains 1..*
    ErrorResponse *-- ErrorBody : contains 1

    %% Server owns types
    ApiServer ..> CompletionRequest : receives
    ApiServer ..> CompletionResponse : produces
    ApiServer ..> CompletionChunk : streams
    ApiServer ..> ErrorResponse : returns on error

    %% Channel bridge
    ApiServer ..> InferenceRequest : sends via channel
    InferenceRequest ..> InferenceResponse : receives via response_tx
```
**Figure 15.1** — API server components. The server receives OpenAI-format requests, validates them, and bridges to the inference engine via channels. Responses are either a single CompletionResponse (non-streaming) or a series of CompletionChunks (streaming).

## Async/Sync Bridge Data Flow

```mermaid
flowchart LR
    subgraph "Async World (HTTP)"
        A["HTTP Client"]
        B["API Handler<br/>(async)"]
        G["SSE Stream<br/>or JSON Response"]
    end

    subgraph "Channel Bridge"
        C["request_tx →<br/>request_rx"]
        F["response_tx →<br/>response_rx"]
    end

    subgraph "Sync World (Inference)"
        D["Engine Thread<br/>(sync)"]
        E["Model Forward Pass<br/>+ Sampling"]
    end

    A -->|"POST /v1/completions"| B
    B -->|"InferenceRequest"| C
    C -->|"recv()"| D
    D --> E
    E -->|"token"| D
    D -->|"InferenceResponse::Token"| F
    F -->|"recv()"| B
    B --> G
    G -->|"data: {...}"| A
```
**Figure 15.2** — The async/sync bridge. HTTP handlers live in the async world; the inference engine runs on a sync thread. Channels connect them: one shared channel for submitting requests, one per-request channel for streaming tokens back.

## Request Lifecycle

```mermaid
flowchart TB
    A["Client sends<br/>POST /v1/completions"] --> B{"Validate<br/>request"}
    B -->|"invalid"| C["400 Bad Request<br/>OpenAI error format"]
    B -->|"valid"| D["Create per-request<br/>response channel"]
    D --> E["Build InferenceRequest<br/>with response_tx"]
    E --> F["Send via<br/>request_tx"]
    F --> G["Engine thread<br/>receives request"]
    G --> H["Run inference loop"]
    H --> I{"More tokens?"}
    I -->|"yes"| J["Send Token<br/>via response_tx"]
    J --> K{"stream?"}
    K -->|"yes"| L["Emit SSE event<br/>data: {...}"]
    L --> I
    K -->|"no"| M["Collect in buffer"]
    M --> I
    I -->|"no"| N["Send Done<br/>via response_tx"]
    N --> O{"stream?"}
    O -->|"yes"| P["Emit data: [DONE]"]
    O -->|"no"| Q["Return full<br/>CompletionResponse JSON"]
```
**Figure 15.3** — Full request lifecycle from client to response. The request is validated, sent through the channel bridge, processed by the engine, and tokens flow back either as SSE events (streaming) or collected into a single JSON response (non-streaming).
