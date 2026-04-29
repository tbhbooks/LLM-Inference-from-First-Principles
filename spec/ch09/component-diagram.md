# Chapter 9: Component Diagram

## Memory Pool Simulator — Class Structure

```mermaid
classDiagram
    class MemoryPool {
        +int total_slots
        +int[] slots
        +allocate_contiguous(request_id, num_slots) (int, bool)
        +deallocate(request_id)
        +utilization() float
        +internal_waste(allocated, actual) int
        +largest_free_block() int
        +total_free() int
        +num_fragments() int
        +visualize(width) string
    }

    class Request {
        +int request_id
        +int prompt_len
        +int max_seq_len
    }

    class SimulationResult {
        +int step
        +string action
        +int request_id
        +bool success
        +float utilization
        +int internal_waste
        +int total_free
        +int largest_free_block
        +int num_fragments
    }

    MemoryPool --> Request : allocates for
    MemoryPool --> SimulationResult : produces
```
**Figure 9.1** — Memory pool simulator components. The pool manages slot-level allocation; requests describe what each sequence needs; results capture the state after each operation.

## Fragmentation Types

```mermaid
graph TB
    subgraph "Internal Fragmentation"
        A1["Request A<br/>needs 10 tokens"] --> A2["Allocated 30 slots<br/>(max_seq_len)"]
        A2 --> A3["20 slots wasted<br/>inside the allocation"]
    end

    subgraph "External Fragmentation"
        B1["45 free slots<br/>scattered across pool"] --> B2["Largest contiguous<br/>block = 15"]
        B2 --> B3["Request needs 25<br/>contiguous slots"]
        B3 --> B4["ALLOCATION FAILS<br/>despite free memory"]
    end

    subgraph "Paged Solution (Ch 10)"
        C1["Request needs<br/>10 tokens"] --> C2["Allocate 1 block<br/>(16 tokens)"]
        C2 --> C3["< 1 block wasted<br/>blocks non-contiguous"]
    end
```
**Figure 9.2** — The two types of memory fragmentation and how paging solves both. Internal waste comes from over-allocation; external waste comes from scattered free space.

## The OS Analogy

```mermaid
graph LR
    subgraph "OS Virtual Memory"
        OS1["Process requests<br/>memory"] --> OS2["OS allocates<br/>pages (4 KB)"]
        OS2 --> OS3["Page table maps<br/>virtual → physical"]
        OS3 --> OS4["Pages can be<br/>anywhere in RAM"]
    end

    subgraph "Paged KV Cache"
        KV1["Request needs<br/>KV storage"] --> KV2["Allocator assigns<br/>blocks (16 tokens)"]
        KV2 --> KV3["Block table maps<br/>logical → physical"]
        KV3 --> KV4["Blocks can be<br/>anywhere in VRAM"]
    end

    OS4 -.->|"same idea"| KV4
```
**Figure 9.3** — The operating system solved memory fragmentation with paging. PagedAttention applies the same idea to KV caches: fixed-size blocks, a mapping table, and non-contiguous allocation.
