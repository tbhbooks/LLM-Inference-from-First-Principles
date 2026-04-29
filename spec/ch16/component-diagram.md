# Chapter 16 -- Component Diagram: Prefix Caching

## PrefixCache Structure

```mermaid
classDiagram
    direction TB

    class PrefixCache {
        cached_blocks: HashMap~BlockHash, BlockId~
        free_queue: VecDeque~BlockId~
        block_size: int
        total_lookups: int
        total_hits: int
        +get_computed_blocks(token_ids) (list~BlockId~, int)
        +cache_full_blocks(block_ids, token_ids)
        +evict() Option~BlockId~
        +hit_rate() float
    }

    class BlockHash {
        <<type alias>>
        value: uint64
    }
    note for BlockHash "Content-addressed:<br/>hash(tokens, prev_hash)"

    class BlockAllocator {
        <<trait>>
        +allocate() BlockId
        +free(block_id)
        +can_allocate(n) bool
        +num_free_blocks() int
    }

    class FreeListAllocator {
        blocks: list~Block~
        free_pool: VecDeque~BlockId~
        +allocate() BlockId
        +free(block_id)
        +increment_ref(block_id)
    }

    class Block {
        block_id: BlockId
        block_size: int
        ref_count: int
        num_filled: int
    }

    class BlockTable {
        block_ids: list~BlockId~
        block_size: int
        +append_block(block_id)
        +slot_for_token(pos) (BlockId, int)
    }

    %% Trait implementation
    FreeListAllocator ..|> BlockAllocator : implements

    %% Composition
    PrefixCache --> BlockHash : uses as key
    PrefixCache --> FreeListAllocator : consults for eviction
    FreeListAllocator o-- Block : manages 0..*
    BlockTable o-- Block : references 0..*
```
**Figure 16.1** — PrefixCache class structure. The cache maps BlockHash values to physical BlockIds. It works alongside the FreeListAllocator (from ch10), adding content-addressed lookup and LRU eviction. Block ref_count enables sharing between sequences.

## Hash Chaining Flow

```mermaid
flowchart LR
    subgraph "Block 0"
        T0["Tokens<br/>[100..115]"]
    end

    subgraph "Block 1"
        T1["Tokens<br/>[116..131]"]
    end

    subgraph "Block 2"
        T2["Tokens<br/>[132..147]"]
    end

    subgraph "Block 3"
        T3["Tokens<br/>[148..163]"]
    end

    H0["hash(tokens_0, None)<br/>= 0xa3f7..."]
    H1["hash(tokens_1, H0)<br/>= 0x1b4e..."]
    H2["hash(tokens_2, H1)<br/>= 0x7c2a..."]
    H3["hash(tokens_3, H2)<br/>= 0xe8d1..."]

    T0 --> H0
    T1 --> H1
    T2 --> H2
    T3 --> H3

    H0 -->|prev_hash| H1
    H1 -->|prev_hash| H2
    H2 -->|prev_hash| H3

    subgraph "Lookup"
        L["cached_blocks<br/>HashMap"]
    end

    H0 -->|lookup| L
    H1 -->|lookup| L
    H2 -->|lookup| L
    H3 -->|lookup| L
```
**Figure 16.2** — Hash chaining flow. Each block's hash incorporates the previous block's hash, creating a chain. This ensures that identical tokens in different contexts produce different hashes. All hashes are looked up in the cached_blocks HashMap.

## Cache Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Free : Pool initialized
    Free --> Allocated : allocate()
    Allocated --> Cached : cache_full_blocks()<br/>ref_count >= 1
    Cached --> SharedCached : cache hit<br/>ref_count += 1
    SharedCached --> Cached : one sequence finishes<br/>ref_count -= 1
    Cached --> FreeCached : last sequence finishes<br/>ref_count = 0
    FreeCached --> Cached : cache hit<br/>ref_count = 1
    FreeCached --> Free : evict()
    Free --> Allocated : allocate()
```
**Figure 16.3** — Cache lifecycle for a physical block. After allocation and fill, blocks enter the cached state. When ref_count drops to zero, blocks become FreeCached: still in the hash table (can serve hits) but eligible for eviction. Only evict() truly returns a block to the free pool.
