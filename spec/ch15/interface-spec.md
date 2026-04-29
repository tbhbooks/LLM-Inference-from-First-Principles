# Chapter 15 -- Interface Specification: Building the API Server

This is a language-agnostic specification. It defines the contracts for
an OpenAI-compatible HTTP API with SSE streaming, and the async/sync
channel bridge that connects it to the inference engine.

---

## 1. Overview

Chapters 1-14 built the complete inference pipeline: model loading, KV cache,
PagedAttention, continuous batching, scheduling, and sampling. But the engine
has no front door. There is no way for a client to send a request and get a
response over the network.

This chapter adds the API layer:

- **OpenAI-compatible endpoints** -- `/v1/completions` and `/v1/chat/completions`
  with the same request/response format that OpenAI clients already speak.
- **SSE streaming** -- Token-by-token delivery using Server-Sent Events, so
  clients see output as it is generated, not after the entire sequence finishes.
- **Channel bridge** -- An async HTTP handler sends requests through a channel
  to a sync engine thread, which sends tokens back through a per-request
  response channel.
- **Error handling** -- OpenAI-format error responses for validation failures
  and internal errors.

After this chapter, the engine can serve real HTTP clients.

---

## 2. Dependencies

From Chapter 13: Engine loop that accepts requests and produces tokens.
From Chapter 14: `SamplingParams` struct with temperature, top-k, top-p,
repetition penalty, and validation rules.

---

## 3. OpenAI Request Types

### 3.1 CompletionRequest

```
struct CompletionRequest:
    model: string              // required, non-empty
    prompt: string             // required, non-empty
    max_tokens: int            // optional, default 16
    temperature: float         // optional, default 1.0
    top_k: int                 // optional, default -1 (disabled)
    top_p: float               // optional, default 1.0
    repetition_penalty: float  // optional, default 1.0
    stream: bool               // optional, default false
    stop: list[string]         // optional, default empty
    user: string               // optional, ignored
```

### 3.2 ChatCompletionRequest

```
struct ChatCompletionRequest:
    model: string              // required, non-empty
    messages: list[Message]    // required, non-empty
    max_tokens: int            // optional, default 16
    temperature: float         // optional, default 1.0
    top_k: int                 // optional, default -1 (disabled)
    top_p: float               // optional, default 1.0
    repetition_penalty: float  // optional, default 1.0
    stream: bool               // optional, default false

struct Message:
    role: string               // "system", "user", or "assistant"
    content: string
```

For the MVP, chat messages are concatenated into a single prompt string
(with role prefixes). A production implementation would use model-specific
chat templates.

---

## 4. OpenAI Response Types

### 4.1 CompletionResponse (non-streaming)

```
struct CompletionResponse:
    id: string                 // "cmpl-" + unique id (e.g., "cmpl-abc123")
    object: string             // "text_completion"
    created: int               // unix timestamp (seconds since epoch)
    model: string              // echo back the requested model name
    choices: list[Choice]
    usage: Usage

struct Choice:
    text: string               // the generated text
    index: int                 // 0 for single completion
    finish_reason: string      // "length" or "stop"

struct Usage:
    prompt_tokens: int         // number of tokens in the prompt
    completion_tokens: int     // number of tokens generated
    total_tokens: int          // prompt_tokens + completion_tokens
```

### 4.2 CompletionChunk (streaming)

```
struct CompletionChunk:
    id: string                 // same id for all chunks in a request
    object: string             // "text_completion"
    created: int               // same timestamp for all chunks
    choices: list[ChunkChoice]

struct ChunkChoice:
    text: string               // a single token's text
    index: int                 // 0
    finish_reason: string?     // null for intermediate tokens, "length" or "stop" for the last
```

### 4.3 ChatCompletionResponse

```
struct ChatCompletionResponse:
    id: string                 // "chatcmpl-" + unique id
    object: string             // "chat.completion"
    created: int
    model: string
    choices: list[ChatChoice]
    usage: Usage

struct ChatChoice:
    message: Message           // role = "assistant", content = generated text
    index: int
    finish_reason: string
```

---

## 5. SSE Streaming Protocol

When `stream: true` is set in the request, the response is delivered as
Server-Sent Events instead of a single JSON body.

### 5.1 HTTP Headers

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

### 5.2 Event Format

Each generated token is sent as one SSE event:

```
data: {"id":"cmpl-abc123","object":"text_completion","created":1234567890,"choices":[{"text":" token","index":0,"finish_reason":null}]}\n\n
```

Rules:
- Each event starts with `data: ` (note the space after the colon).
- The payload is a JSON-serialized `CompletionChunk`.
- Each event is terminated by two newlines (`\n\n`).
- `finish_reason` is `null` for all intermediate tokens.
- The last real token has `finish_reason` set to `"length"` or `"stop"`.

### 5.3 Done Sentinel

After the last token, send the sentinel event:

```
data: [DONE]\n\n
```

This tells the client the stream is complete. The `[DONE]` string is a
literal, not JSON.

---

## 6. Channel Bridge

The API server runs async (handling HTTP connections), but the inference engine
runs sync (computing forward passes). Channels bridge these two worlds.

### 6.1 Channel Types

```
// Shared across all requests — one sender per handler, one receiver for the engine
request_tx: Sender<InferenceRequest>
request_rx: Receiver<InferenceRequest>

// Per-request — created by the handler, passed to the engine inside the request
response_tx: Sender<InferenceResponse>
response_rx: Receiver<InferenceResponse>
```

### 6.2 InferenceRequest

```
struct InferenceRequest:
    request_id: string                     // unique per request
    prompt: string                         // the text to complete
    sampling_params: SamplingParams        // from ch14
    max_tokens: int                        // how many tokens to generate
    response_tx: Sender<InferenceResponse> // channel to send results back
```

### 6.3 InferenceResponse

```
enum InferenceResponse:
    Token { text: string, finish_reason: Option<string> }
    Error { message: string }
    Done
```

### 6.4 Bridge Flow

1. HTTP handler receives a `CompletionRequest`.
2. Handler creates a per-request channel: `(response_tx, response_rx)`.
3. Handler builds an `InferenceRequest` with the `response_tx` inside it.
4. Handler sends the `InferenceRequest` via the shared `request_tx`.
5. Engine thread receives the request via `request_rx`.
6. Engine runs inference, sending `InferenceResponse::Token` for each token.
7. Engine sends `InferenceResponse::Done` when generation is complete.
8. If `stream: true`: handler reads from `response_rx` and emits each token
   as an SSE event immediately.
9. If `stream: false`: handler collects all tokens from `response_rx`, then
   returns a single `CompletionResponse` JSON.

---

## 7. API Router

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/v1/completions` | `handle_completion` | Text completion |
| POST | `/v1/chat/completions` | `handle_chat_completion` | Chat completion |
| GET | `/v1/models` | `handle_list_models` | List available models |
| GET | `/health` | `handle_health` | Health check |

### 7.1 /health

```
GET /health → 200 OK
{"status": "ok"}
```

No authentication, no request body. Always returns 200 if the server is up.

### 7.2 /v1/models

```
GET /v1/models → 200 OK
{
  "object": "list",
  "data": [
    {
      "id": "<model-name>",
      "object": "model",
      "created": <unix-timestamp>,
      "owned_by": "rvllm"
    }
  ]
}
```

### 7.3 /v1/completions

```
POST /v1/completions
Content-Type: application/json

{"model": "gpt2", "prompt": "The future of AI is", "max_tokens": 5}

→ 200 OK (see CompletionResponse in section 4.1)
```

### 7.4 /v1/chat/completions

```
POST /v1/chat/completions
Content-Type: application/json

{"model": "gpt2", "messages": [{"role": "user", "content": "Hello"}]}

→ 200 OK (see ChatCompletionResponse in section 4.3)
```

---

## 8. Request Validation

All validation errors return HTTP 400 with the OpenAI error format (section 9).

| Field | Rule | Error message |
|-------|------|---------------|
| `model` | non-empty string | "model is required" |
| `prompt` | non-empty string (completions) | "prompt is required" |
| `messages` | non-empty list (chat) | "messages is required" |
| `max_tokens` | >= 1 | "max_tokens must be >= 1" |
| `temperature` | >= 0.0 | "temperature must be non-negative" |
| `top_k` | -1, or >= 1 | "top_k must be -1 (disabled) or >= 1" |
| `top_p` | 0.0 < p <= 1.0 | "top_p must be in (0.0, 1.0]" |
| `repetition_penalty` | >= 1.0 | "repetition_penalty must be >= 1.0" |

These rules match the SamplingParams validation from Chapter 14.

---

## 9. Error Response Format

All errors use the OpenAI error object format:

```
{
  "error": {
    "message": "description of what went wrong",
    "type": "invalid_request_error",
    "code": null
  }
}
```

### 9.1 Error Types

| HTTP Status | `type` field | When |
|-------------|-------------|------|
| 400 | `"invalid_request_error"` | Missing required fields, invalid parameter values |
| 404 | `"invalid_request_error"` | Unknown endpoint or model |
| 500 | `"server_error"` | Engine failure, channel disconnected, unexpected error |

---

## 10. Demo Program

The demo (`examples/ch15_api_server`) simulates the full API lifecycle
without starting a real HTTP server or loading a real model. It exercises
the types, validation, channel bridge, and SSE formatting in-process.

### 10.1 Part 1: Server Startup

Print the server configuration as it would appear at startup:

```
=== PART 1: Server Startup ===
Starting rvllm server on http://127.0.0.1:8080
Model: mock-gpt2
Endpoints:
  POST /v1/completions
  POST /v1/chat/completions
  GET  /v1/models
  GET  /health
```

### 10.2 Part 2: Health Check

Simulate a health check request and response:

```
=== PART 2: Health Check ===
GET /health → 200 OK
{"status": "ok"}
```

### 10.3 Part 3: Completion Request

Simulate a non-streaming completion request. Show the request JSON and the
full response JSON including id, object, created, model, choices, and usage:

```
=== PART 3: Completion Request ===
POST /v1/completions
Request: {"model": "mock-gpt2", "prompt": "The future of AI is", "max_tokens": 5, "stream": false}

Response:
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": ...,
  "model": "mock-gpt2",
  "choices": [
    {
      "text": "...",
      "index": 0,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 5,
    "total_tokens": 10
  }
}
```

### 10.4 Part 4: Streaming Response

Simulate a streaming completion request. Show each SSE event as it would
be sent over the wire:

```
=== PART 4: Streaming Response ===
POST /v1/completions (stream=true)

data: {"id":"cmpl-...","object":"text_completion","created":...,"choices":[{"text":" bright","index":0,"finish_reason":null}]}

data: {"id":"cmpl-...","object":"text_completion","created":...,"choices":[{"text":" and","index":0,"finish_reason":null}]}

data: {"id":"cmpl-...","object":"text_completion","created":...,"choices":[{"text":" full","index":0,"finish_reason":null}]}

data: {"id":"cmpl-...","object":"text_completion","created":...,"choices":[{"text":" of","index":0,"finish_reason":null}]}

data: {"id":"cmpl-...","object":"text_completion","created":...,"choices":[{"text":" promise","index":0,"finish_reason":"length"}]}

data: [DONE]
```

### 10.5 Part 5: Error Handling

Simulate an invalid request (missing required field) and show the error
response in OpenAI format:

```
=== PART 5: Error Handling ===
POST /v1/completions (missing prompt)
→ 400 Bad Request

{
  "error": {
    "message": "prompt is required",
    "type": "invalid_request_error",
    "code": null
  }
}
```

---

## 11. Correctness Criteria

1. **Endpoint routing:** All four endpoints are defined and reachable.
2. **OpenAI format:** Response JSON matches the OpenAI API structure
   (id, object, created, choices, usage).
3. **SSE protocol:** Streaming responses use `data: ` prefix, JSON payload,
   double-newline termination, and `[DONE]` sentinel.
4. **Channel bridge:** Requests flow from HTTP handler through a channel to
   the engine thread; responses flow back through a per-request channel.
5. **Validation:** Missing required fields and out-of-range parameters
   produce HTTP 400 with the OpenAI error format.
6. **Error format:** All errors include `message`, `type`, and `code` fields
   inside an `error` wrapper.
7. **Health endpoint:** Returns `{"status": "ok"}` with HTTP 200.
8. **Completion marker:** Output ends with "Chapter 15 complete."

---

## 12. Validation Summary

| Test | What it checks |
|------|---------------|
| Part 1 present | "Server Startup" section exists |
| Part 2 present | "Health Check" section exists |
| Part 3 present | "Completion Request" section exists |
| Part 4 present | "Streaming Response" section exists |
| Part 5 present | "Error Handling" section exists |
| Server config | Shows address/port and endpoint list |
| Health response | Shows `{"status": "ok"}` |
| Completion format | Shows OpenAI response fields (id, object, choices, usage) |
| SSE format | Shows `data: ` prefix and `[DONE]` sentinel |
| Channel bridge | Mentions channel, sender, or receiver |
| Error format | Shows OpenAI error object with message and type |
| Completion marker | "Chapter 15 complete" appears |
