# Chapter 15 -- Sequence Diagram: Building the API Server

## Diagram 1: Streaming Completion Request

```mermaid
sequenceDiagram
    participant Client as HTTP Client
    participant Handler as API Handler
    participant ReqCh as Request Channel
    participant Engine as Engine Thread
    participant RespCh as Response Channel

    Client->>Handler: POST /v1/completions<br/>(stream=true)

    Handler->>Handler: Validate CompletionRequest

    Note over Handler: Create per-request<br/>response channel

    Handler->>ReqCh: send(InferenceRequest<br/>+ response_tx)

    Note over Handler: Start SSE stream<br/>Set headers:<br/>Content-Type: text/event-stream

    ReqCh->>Engine: recv() → InferenceRequest

    Note over Engine: Run inference loop

    loop For each generated token
        Engine->>RespCh: send(Token(" bright", null))
        RespCh->>Handler: recv() → Token
        Handler->>Client: data: {"choices":[{"text":" bright"}]}
    end

    Engine->>RespCh: send(Token(" end", "length"))
    RespCh->>Handler: recv() → Token (finish_reason="length")
    Handler->>Client: data: {"choices":[{"text":" end","finish_reason":"length"}]}

    Engine->>RespCh: send(Done)
    RespCh->>Handler: recv() → Done
    Handler->>Client: data: [DONE]
```
**Figure 15.4** — Streaming completion lifecycle. The handler opens an SSE stream, then forwards each token from the response channel as a `data:` event. The `[DONE]` sentinel signals end-of-stream.

## Diagram 2: Non-Streaming Completion Request

```mermaid
sequenceDiagram
    participant Client as HTTP Client
    participant Handler as API Handler
    participant ReqCh as Request Channel
    participant Engine as Engine Thread
    participant RespCh as Response Channel

    Client->>Handler: POST /v1/completions<br/>(stream=false)

    Handler->>Handler: Validate CompletionRequest

    Note over Handler: Create per-request<br/>response channel

    Handler->>ReqCh: send(InferenceRequest<br/>+ response_tx)

    ReqCh->>Engine: recv() → InferenceRequest

    Note over Engine: Run inference loop

    loop Collect all tokens
        Engine->>RespCh: send(Token(" bright", null))
        RespCh->>Handler: recv() → Token
        Note over Handler: Append to buffer
    end

    Engine->>RespCh: send(Token(" end", "length"))
    RespCh->>Handler: recv() → Token
    Note over Handler: Append final token

    Engine->>RespCh: send(Done)
    RespCh->>Handler: recv() → Done

    Note over Handler: Build CompletionResponse<br/>with all collected text,<br/>compute usage counts

    Handler->>Client: 200 OK<br/>CompletionResponse JSON
```
**Figure 15.5** — Non-streaming completion. The handler collects all tokens from the response channel into a buffer, then returns a single CompletionResponse JSON with the full text, choices, and usage.

## Diagram 3: Error Handling Flow

```mermaid
sequenceDiagram
    participant Client as HTTP Client
    participant Handler as API Handler
    participant Engine as Engine Thread

    rect rgb(255, 235, 235)
        Note over Client,Handler: Validation error (400)
        Client->>Handler: POST /v1/completions<br/>(missing prompt)
        Handler->>Handler: Validate → missing "prompt"
        Handler->>Client: 400 Bad Request<br/>{"error":{"message":"prompt is required",<br/>"type":"invalid_request_error","code":null}}
    end

    rect rgb(255, 245, 230)
        Note over Client,Engine: Engine error (500)
        Client->>Handler: POST /v1/completions<br/>(valid request)
        Handler->>Handler: Validate → OK
        Handler->>Engine: send(InferenceRequest)
        Engine->>Handler: InferenceResponse::Error("engine failure")
        Handler->>Client: 500 Internal Server Error<br/>{"error":{"message":"engine failure",<br/>"type":"server_error","code":null}}
    end

    rect rgb(235, 245, 255)
        Note over Client,Handler: Unknown endpoint (404)
        Client->>Handler: GET /v1/unknown
        Handler->>Client: 404 Not Found<br/>{"error":{"message":"not found",<br/>"type":"invalid_request_error","code":null}}
    end
```
**Figure 15.6** — Error handling paths. Validation failures return 400 before reaching the engine. Engine errors return 500. Unknown endpoints return 404. All errors use the OpenAI error format.
