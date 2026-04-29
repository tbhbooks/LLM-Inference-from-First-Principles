# Chapter 16 -- Sequence Diagram: Prefix Caching

## Diagram 1: Cache Miss Then Cache Hit

```mermaid
sequenceDiagram
    participant Sched as Scheduler
    participant PC as PrefixCache
    participant Alloc as BlockAllocator
    participant Engine as Engine

    Note over Sched: Request A arrives: 80 tokens<br/>(64 prefix + 16 unique)

    Sched->>PC: get_computed_blocks([100..179])
    Note over PC: Block 0: hash(tokens, None)<br/>Lookup → MISS
    PC-->>Sched: ([], 0 computed tokens)

    Sched->>Alloc: allocate() x 5
    Alloc-->>Sched: blocks [0, 1, 2, 3, 4]

    Sched->>Engine: prefill(80 tokens, all 5 blocks)
    Note over Engine: Compute KV for all 80 tokens

    Sched->>PC: cache_full_blocks([0,1,2,3,4], tokens)
    Note over PC: Insert 5 hashes into cached_blocks

    Note over Sched: Request B arrives: 80 tokens<br/>(same 64 prefix + 16 different)

    Sched->>PC: get_computed_blocks([100..179'])
    Note over PC: Block 0: hash → HIT (block 0)<br/>Block 1: hash → HIT (block 1)<br/>Block 2: hash → HIT (block 2)<br/>Block 3: hash → HIT (block 3)<br/>ref_count 1 → 2 on each
    PC-->>Sched: ([0,1,2,3], 64 computed tokens)

    Sched->>Alloc: allocate() x 1
    Alloc-->>Sched: block [5]

    Sched->>Engine: prefill(16 tokens only, skip 64)
    Note over Engine: Compute KV for 16 tokens<br/>64 tokens FREE — already cached
```
**Figure 16.4** — Cache miss then cache hit. Request A populates the prefix cache. Request B with the same prefix gets 4 block hits and only needs to compute KV for 16 new tokens instead of 80.

## Diagram 2: Eviction Flow

```mermaid
sequenceDiagram
    participant Req as Request
    participant PC as PrefixCache
    participant FQ as free_queue
    participant HT as cached_blocks

    Note over Req: Request A finishes

    Req->>PC: free blocks [0, 1, 2, 3, 4]

    loop For each block
        PC->>PC: ref_count -= 1
        alt ref_count == 0
            PC->>FQ: push_back(block_id)
            Note over FQ: Block stays in cached_blocks<br/>but is eviction-eligible
        end
    end

    Note over FQ: free_queue = [4, 0, 1, 2, 3]<br/>All ref_count = 0

    Note over Req: New request needs a block<br/>but free pool is empty

    Req->>PC: evict()
    PC->>FQ: pop_front() → block 4 (oldest)
    PC->>HT: remove(hash_of_block_4)
    Note over HT: Block 4 hash deleted<br/>No future cache hits for it
    PC-->>Req: block 4 (truly free now)
```
**Figure 16.5** — Eviction flow. When a request finishes, its blocks enter the free_queue but remain in cached_blocks. When memory pressure requires a new block, evict() pops the oldest from free_queue and removes it from the hash table.

## Diagram 3: ref_count Lifecycle

```mermaid
sequenceDiagram
    participant A as Request A
    participant B as Request B
    participant C as Request C
    participant Blk as Block 0

    Note over Blk: ref_count = 0 (free)

    A->>Blk: allocate()
    Note over Blk: ref_count = 1

    A->>Blk: cache_full_blocks()
    Note over Blk: Now in cached_blocks<br/>ref_count = 1

    B->>Blk: get_computed_blocks() → HIT
    Note over Blk: ref_count = 2<br/>Shared by A and B

    A->>Blk: finish() → ref_count -= 1
    Note over Blk: ref_count = 1<br/>Still active for B

    B->>Blk: finish() → ref_count -= 1
    Note over Blk: ref_count = 0<br/>Enters free_queue<br/>Still in cached_blocks

    C->>Blk: get_computed_blocks() → HIT
    Note over Blk: ref_count = 1<br/>Removed from free_queue<br/>Reused without eviction!

    C->>Blk: finish() → ref_count -= 1
    Note over Blk: ref_count = 0<br/>Back in free_queue
```
**Figure 16.6** — ref_count lifecycle for a shared block. Request A allocates, Request B shares via cache hit, both finish, and Request C reuses the freed-but-cached block without eviction. The block transitions between active and free-cached states based on ref_count.
