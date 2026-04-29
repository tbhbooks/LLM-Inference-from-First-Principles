# Chapter 3 -- Sequence Diagrams

## CLI Dispatch Flow

The binary parses command-line arguments, matches on the subcommand, and
dispatches. At this stage every path hits a stub that prints a message and
exits cleanly.

```mermaid
sequenceDiagram
    actor User
    participant CLI as CLI (main)
    participant ArgParser as Argument Parser
    participant GenHandler as generate handler
    participant InspectHandler as inspect handler

    User ->> CLI: rvllm generate --prompt "Hello"
    CLI ->> ArgParser: parse(args)
    ArgParser -->> CLI: Commands::Generate { model, prompt, max_tokens }
    CLI ->> GenHandler: dispatch(model, prompt, max_tokens)
    Note right of GenHandler: (stub) No model loading,<br/>no tokenization,<br/>no inference yet
    GenHandler -->> CLI: print "generate: not yet implemented"
    CLI -->> User: exit 0
```

```mermaid
sequenceDiagram
    actor User
    participant CLI as CLI (main)
    participant ArgParser as Argument Parser
    participant InspectHandler as inspect handler

    User ->> CLI: rvllm inspect --model gpt2
    CLI ->> ArgParser: parse(args)
    ArgParser -->> CLI: Commands::Inspect { model }
    CLI ->> InspectHandler: dispatch(model)
    Note right of InspectHandler: (stub) No model loading,<br/>no config display yet
    InspectHandler -->> CLI: print "inspect: not yet implemented"
    CLI -->> User: exit 0
```

## Future Flow (what the stubs will become)

This diagram shows the full pipeline that later chapters will fill in.
Chapter 3 establishes the module boundaries; the arrows below show where
data will eventually flow.

```mermaid
sequenceDiagram
    actor User
    participant CLI as CLI (main)
    participant Config as config
    participant Tok as tokenizer
    participant Engine as engine
    participant Sched as scheduler
    participant Mdl as model
    participant Mem as memory
    participant Samp as sampling

    User ->> CLI: rvllm generate --prompt "Hello" --max-tokens 64
    CLI ->> Config: build EngineConfig from flags
    CLI ->> Tok: load tokenizer for model
    Tok -->> CLI: TokenizerBackend instance

    CLI ->> Engine: new(config, tokenizer, model, sampler)
    CLI ->> Engine: generate(prompt, max_tokens)

    Engine ->> Tok: encode(prompt)
    Tok -->> Engine: list<TokenId>

    loop until max_tokens or EOS
        Engine ->> Sched: next_batch()
        Sched -->> Engine: batch of sequences
        Engine ->> Mdl: forward(input_ids, pos)
        Mdl ->> Mem: read/write KV cache blocks
        Mdl -->> Engine: logits Tensor
        Engine ->> Samp: sample(logits)
        Samp -->> Engine: TokenId
    end

    Engine ->> Tok: decode(generated_ids)
    Tok -->> Engine: output text
    Engine -->> CLI: output text
    CLI -->> User: print output
```

## Error Propagation

Any module can fail. Errors propagate upward through `Result`, converted
into `RvllmError` variants, until the CLI catches them and prints a
user-facing message.

```mermaid
sequenceDiagram
    participant Module as Any Module
    participant Error as RvllmError
    participant CLI as CLI (main)
    actor User

    Module ->> Error: return Err(RvllmError::variant(detail))
    Error -->> CLI: propagated via Result / ? operator
    CLI ->> CLI: match on RvllmError variant
    CLI -->> User: print human-readable error message
    CLI -->> User: exit non-zero
```
