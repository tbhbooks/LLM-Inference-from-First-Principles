# Chapter 15 -- LLM Prompt Template: Building the API Server

Copy and paste the prompt below into your LLM of choice, or use it as a
reference while implementing by hand. This builds on the project from
Chapters 1-14.

---

## The Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 15.
I have an existing project from Chapters 1-14 with:
- Full GPT-2 model with KV cache (ch04-06)
- Greedy generation loop (ch07)
- PagedAttention memory management (ch09-10)
- Continuous batching scheduler (ch11-12)
- Engine loop with Sampler trait (ch13)
- Full sampling pipeline: temperature, top-k, top-p, repetition penalty (ch14)

Now wrap the inference engine in an OpenAI-compatible HTTP API with SSE
streaming. Bridge the async HTTP world and the sync inference world using
channels.

LANGUAGE/FRAMEWORK: [Rust with axum / Python with FastAPI / your choice]

=== WHAT TO CREATE / MODIFY ===

  NEW FILES:
    src/api/mod.rs          <-- Router setup, handler functions, server startup
    src/api/types.rs        <-- OpenAI request/response types (CompletionRequest, etc.)
    src/api/sse.rs          <-- SSE streaming utilities
    examples/ch15_api_server <-- Demo program (mock server, no real model)

  MODIFY:
    src/engine/mod.rs       <-- Channel integration (request_tx/request_rx)
    src/main.rs             <-- Add "serve" subcommand

  KEEP UNCHANGED:
    src/model/ (all files)
    src/tokenizer/ (all files)
    src/sampling/ (all files)
    src/scheduler/ (all files)
    src/memory/ (all files)

=== OPENAI COMPLETION REQUEST ===

  struct CompletionRequest:
      model: string              // required
      prompt: string             // required for /v1/completions
      max_tokens: int            // optional, default 16
      temperature: float         // optional, default 1.0
      top_k: int                 // optional, default -1 (disabled)
      top_p: float               // optional, default 1.0
      repetition_penalty: float  // optional, default 1.0
      stream: bool               // optional, default false
      stop: list[string]         // optional, default empty
      user: string               // optional, ignored

  Validation:
      model is non-empty
      prompt is non-empty (for /v1/completions)
      max_tokens >= 1
      temperature >= 0.0
      top_k == -1 or top_k >= 1
      0.0 < top_p <= 1.0
      repetition_penalty >= 1.0

=== OPENAI COMPLETION RESPONSE ===

  struct CompletionResponse:
      id: string                 // "cmpl-" + unique id
      object: string             // "text_completion"
      created: int               // unix timestamp
      model: string              // echo back the model name
      choices: list[Choice]
      usage: Usage

  struct Choice:
      text: string               // generated text
      index: int                 // always 0 for single completion
      finish_reason: string      // "length" or "stop" or null (streaming)

  struct Usage:
      prompt_tokens: int
      completion_tokens: int
      total_tokens: int

=== CHAT COMPLETION REQUEST ===

  struct ChatCompletionRequest:
      model: string              // required
      messages: list[Message]    // required
      max_tokens: int            // optional, default 16
      temperature: float         // optional, default 1.0
      top_k: int                 // optional, default -1
      top_p: float               // optional, default 1.0
      repetition_penalty: float  // optional, default 1.0
      stream: bool               // optional, default false

  struct Message:
      role: string               // "system", "user", "assistant"
      content: string

  For the MVP, concatenate messages into a single prompt string.
  A real implementation would use a chat template.

=== SSE STREAMING PROTOCOL ===

  Each token is sent as an SSE event:

      data: {"id":"cmpl-...","object":"text_completion","created":...,"choices":[{"text":" token","index":0,"finish_reason":null}]}\n\n

  Final event:

      data: [DONE]\n\n

  HTTP headers for streaming:
      Content-Type: text/event-stream
      Cache-Control: no-cache
      Connection: keep-alive

  The stream field in the request controls whether to use SSE (true)
  or return a single JSON response (false).

=== CHANNEL BRIDGE ===

  The async/sync bridge uses two channel pairs:

  Channel 1: Request submission
      request_tx: Sender<InferenceRequest>    // HTTP handler sends
      request_rx: Receiver<InferenceRequest>  // Engine thread receives

  Channel 2: Token streaming (per-request)
      response_tx: Sender<InferenceResponse>  // Engine thread sends
      response_rx: Receiver<InferenceResponse> // HTTP handler receives

  struct InferenceRequest:
      request_id: string
      prompt: string
      sampling_params: SamplingParams
      max_tokens: int
      response_tx: Sender<InferenceResponse>  // one-shot or streaming

  enum InferenceResponse:
      Token { text: string, finish_reason: Option<string> }
      Error { message: string }
      Done

  Flow:
  1. HTTP handler creates (response_tx, response_rx)
  2. HTTP handler sends InferenceRequest (with response_tx) via request_tx
  3. Engine thread receives request via request_rx
  4. Engine generates tokens, sends each via response_tx
  5. HTTP handler receives tokens via response_rx
  6. If streaming: each token → SSE event
     If not streaming: collect all tokens → single JSON response
  7. Engine sends Done when finished

=== API ROUTES ===

  POST /v1/completions       → handle_completion
  POST /v1/chat/completions  → handle_chat_completion
  GET  /v1/models            → handle_list_models
  GET  /health               → handle_health

  /health returns: {"status": "ok"}

  /v1/models returns:
  {
    "object": "list",
    "data": [{"id": "model-name", "object": "model", "created": ..., "owned_by": "rvllm"}]
  }

=== ERROR RESPONSE FORMAT ===

  All errors use the OpenAI error format:

  {
    "error": {
      "message": "description of what went wrong",
      "type": "invalid_request_error" | "server_error",
      "code": null
    }
  }

  HTTP status codes:
      400 — Invalid request (missing fields, bad parameter values)
      404 — Unknown endpoint
      500 — Internal engine error

=== DEMO PROGRAM (examples/ch15_api_server) ===

The demo does NOT start a real HTTP server or load a real model. It
simulates the full request lifecycle in-process:

1. Server Startup — Print the server config (address, model, endpoints)
2. Health Check — Simulate GET /health, print response
3. Completion Request — Simulate POST /v1/completions (stream=false),
   show the full JSON response
4. Streaming Response — Simulate POST /v1/completions (stream=true),
   show each SSE event line as it would be sent
5. Error Handling — Simulate an invalid request, show the error response

Output format: 5 parts (Server Startup, Health Check, Completion Request,
Streaming Response, Error Handling).

End with: "Chapter 15 complete. Next: Prefix Caching (ch16)"

=== WHAT SUCCESS LOOKS LIKE ===

Running the demo produces clearly labeled output showing:
- The server would listen on a specific address with specific endpoints
- Health check returns OK
- A completion request produces a valid OpenAI-format JSON response
- A streaming request produces SSE events with "data: " prefix and "[DONE]"
- An invalid request produces an OpenAI-format error response

The output demonstrates that the API layer correctly bridges between
HTTP request/response format and the internal inference engine.

=== WHAT TO PRODUCE ===

Produce the complete set of NEW or MODIFIED source files.
Do NOT recreate files from earlier chapters that are unchanged.

After this chapter, the engine has an OpenAI-compatible API layer with
SSE streaming support — ready to serve real HTTP requests.
```
