# Chapter 1: Component Diagram

## KV Cache Memory Calculator — Class Structure

```mermaid
classDiagram
    class ModelConfig {
        +String name
        +int num_layers
        +int num_heads
        +int head_dim
        +int dtype_bytes
        +kv_vector_bytes() int
        +kv_per_token_per_layer() int
        +kv_per_token() int
        +kv_for_sequence(seq_len: int) int
        +max_concurrent_sequences(seq_len: int, memory_bytes: int) int
    }

    class Helpers {
        +format_bytes(bytes: int) String
        +separator()
        +section(title: String)
    }

    class PresetModels {
        +preset_models() List~ModelConfig~
    }

    class Main {
        +main()
    }

    PresetModels ..> ModelConfig : creates
    Main --> PresetModels : calls
    Main --> ModelConfig : calculates with
    Main --> Helpers : formats with

    note for ModelConfig "Preset configurations:\n- GPT-2 (124M): 12 layers, 12 heads, 64 dim, 2 bytes\n- LLaMA-7B: 32 layers, 32 heads, 128 dim, 2 bytes\n- LLaMA-70B: 80 layers, 64 heads, 128 dim, 2 bytes"
```

## Method Chain (smallest unit to largest)

```mermaid
graph LR
    A["kv_vector_bytes()<br/>num_heads * head_dim * dtype_bytes"] --> B["kv_per_token_per_layer()<br/>2 * kv_vector_bytes()"]
    B --> C["kv_per_token()<br/>kv_per_token_per_layer() * num_layers"]
    C --> D["kv_for_sequence(seq_len)<br/>kv_per_token() * seq_len"]
    D --> E["max_concurrent_sequences(seq_len, memory_bytes)<br/>memory_bytes / kv_for_sequence(seq_len)"]
```

## Preset Models

| Model | num_layers | num_heads | head_dim | dtype_bytes |
|-------|-----------|-----------|----------|-------------|
| GPT-2 (124M) | 12 | 12 | 64 | 2 |
| LLaMA-7B | 32 | 32 | 128 | 2 |
| LLaMA-70B | 80 | 64 | 128 | 2 |
