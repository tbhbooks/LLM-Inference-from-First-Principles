# Chapter 16 -- Interface Specification: Prefix Caching

This is a language-agnostic specification. It defines the contracts for
content-addressed block reuse (prefix caching) so that requests sharing a
common prompt prefix skip redundant KV computation.

---

## 1. Overview

Every chat message starts with a system prompt. Every few-shot example repeats
the same demonstrations. Without prefix caching, the engine recomputes the
same KV entries for every request that shares a common prefix. That is pure
waste.

Prefix caching solves this by hashing the token content of each full block,
storing the mapping from hash to physical block, and reusing blocks on future
requests that produce the same hash chain. The result: the second request
with a 2048-token system prompt pays zero prefill cost for those 2048 tokens.

Key ideas:
- **Hash chaining** — block N's hash = hash(tokens_N, hash_N-1). This prevents
  false hits across different prefix contexts.
- **Only full blocks** — partial blocks are not cached because their content
  may still change.
- **LRU eviction** — freed blocks stay in the cache until memory pressure
  forces eviction of the oldest unused block.
- **ref_count sharing** — multiple sequences can reference the same physical
  block simultaneously.

---

## 2. Dependencies

From Chapter 10:
- `Block` with `block_id`, `block_size`, `ref_count`, `num_filled`
- `BlockAllocator` trait with `allocate()`, `free()`, `can_allocate()`,
  `num_free_blocks()`
- `BlockTable` for per-sequence logical-to-physical mapping

This chapter adds prefix caching on top of the existing block allocator
infrastructure. The allocator itself is modified to support ref_count
increment (for cache hits) and to consult the prefix cache before eviction.

---

## 3. BlockHash and Hash Chaining

### 3.1 BlockHash Type

```
type BlockHash = uint64
```

A content-addressed identifier for a full block of tokens. Two blocks with
identical tokens AND identical prefix history produce the same BlockHash.

### 3.2 hash_block Function

```
hash_block(tokens: list[TokenId], prev_hash: Option[BlockHash]) -> BlockHash
```

Computes the content hash for a single block. The hash incorporates:
1. The previous block's hash (if any) — this is the "chain"
2. Every token ID in the block

```
function hash_block(tokens, prev_hash):
    state = hash_init()                        // start a new hash state
    if prev_hash is not None:
        state = hash_update(state, prev_hash.to_bytes())  // chain to predecessor
    for token in tokens:
        state = hash_update(state, token.to_bytes())      // fold in each token
    return hash_finalize(state)                // produce uint64
```

Any standard hash function works (FNV-1a, SipHash, xxHash, etc.). The choice
does not affect correctness, only collision probability.

### 3.3 Why Chaining Matters

Without chaining, two blocks containing `[50, 51, 52, ..., 65]` would hash
identically regardless of what came before them. That is wrong — the KV values
for those tokens depend on all preceding tokens (attention is causal).

With chaining:
- Block 0: hash = hash([50..65], None)
- Block 1: hash = hash([66..81], hash_of_block_0)

If block 0 changes (different tokens), block 1's hash changes too, even if
block 1's own tokens are identical. No false hits.

### 3.4 Only Full Blocks Are Hashable

A block must have `num_filled == block_size` to be hashed and cached.

Rationale: a partial block's content may grow as more tokens are generated.
Caching it prematurely would create stale entries. Only finalized (full)
blocks have stable content.

---

## 4. PrefixCache Struct

```
struct PrefixCache:
    cached_blocks: HashMap<BlockHash, BlockId>    // hash → physical block
    free_queue: VecDeque<BlockId>                  // LRU eviction order
    block_size: int                                // tokens per block
    // Metrics
    total_lookups: int
    total_hits: int
```

### 4.1 Fields

| Field | Type | Description |
|-------|------|-------------|
| `cached_blocks` | HashMap<BlockHash, BlockId> | Maps content hash to physical block ID |
| `free_queue` | VecDeque<BlockId> | Blocks with ref_count=0, oldest at front |
| `block_size` | int | Tokens per block (must match allocator) |
| `total_lookups` | int | Number of block-level cache lookups performed |
| `total_hits` | int | Number of successful cache hits |

### 4.2 Invariants

- Every BlockId in `cached_blocks` refers to a valid physical block.
- Every BlockId in `free_queue` has `ref_count == 0` and is also in `cached_blocks`.
- A block can be in `cached_blocks` with `ref_count > 0` (actively used) — it
  will NOT be in `free_queue`.
- A block can be in `cached_blocks` with `ref_count == 0` — it IS in `free_queue`
  and eligible for eviction.
- A block NOT in `cached_blocks` is either freshly allocated or already evicted.

---

## 5. get_computed_blocks Algorithm

```
get_computed_blocks(token_ids: list[TokenId]) -> (list[BlockId], num_computed_tokens: int)
```

Walk through the token sequence block by block, hashing and looking up each
full block in the cache. Stop on the first miss.

```
function get_computed_blocks(token_ids):
    result = []
    prev_hash = None
    num_full_blocks = len(token_ids) / block_size    // integer division

    for i in 0..num_full_blocks:
        // Extract this block's tokens
        start = i * block_size
        end = start + block_size
        block_tokens = token_ids[start..end]

        // Compute chained hash
        block_hash = hash_block(block_tokens, prev_hash)
        total_lookups += 1

        // Look up in cache
        if block_hash in cached_blocks:
            block_id = cached_blocks[block_hash]
            total_hits += 1

            // Increment ref_count — this block is now shared
            blocks[block_id].ref_count += 1

            // If block was in free_queue, remove it (it's active again)
            if block_id in free_queue:
                free_queue.remove(block_id)

            result.append(block_id)
            prev_hash = block_hash
        else:
            // STOP on first miss — prefix caching is contiguous only
            break

    num_computed_tokens = len(result) * block_size
    return (result, num_computed_tokens)
```

### 5.1 Why Stop on First Miss

KV computation is sequential — block N's attention output depends on blocks
0..N-1. If block 2 is missing from the cache, we cannot use a cached block 3
even if it exists, because block 3's KV values were computed with a specific
block 2 as context. The prefix must be contiguous.

---

## 6. cache_full_blocks

```
cache_full_blocks(block_ids: list[BlockId], token_ids: list[TokenId])
```

Insert completed full blocks into the prefix cache after prefill.

```
function cache_full_blocks(block_ids, token_ids):
    prev_hash = None
    num_full_blocks = len(token_ids) / block_size    // integer division

    for i in 0..num_full_blocks:
        if i >= len(block_ids):
            break

        start = i * block_size
        end = start + block_size
        block_tokens = token_ids[start..end]

        block_hash = hash_block(block_tokens, prev_hash)

        // Only cache if not already present
        if block_hash not in cached_blocks:
            cached_blocks[block_hash] = block_ids[i]

        prev_hash = block_hash
```

### 6.1 Idempotency

Calling `cache_full_blocks` with blocks that are already cached is a no-op
for those blocks. The existing mapping is preserved.

### 6.2 Partial Block Excluded

If `len(token_ids)` is not a multiple of `block_size`, the last partial block
is not hashed and not inserted into the cache.

---

## 7. Eviction Policy (LRU)

```
evict() -> Option[BlockId]
```

When the allocator needs a free block and the free pool is empty, call evict:

```
function evict():
    while free_queue is not empty:
        block_id = free_queue.pop_front()           // oldest first (LRU)

        // Find and remove from cached_blocks
        // (need reverse lookup: block_id → hash)
        hash_to_remove = None
        for (hash, bid) in cached_blocks:
            if bid == block_id:
                hash_to_remove = hash
                break

        if hash_to_remove is not None:
            cached_blocks.remove(hash_to_remove)

        return Some(block_id)

    return None    // nothing to evict
```

### 7.1 Reverse Lookup Optimization

The naive eviction above scans `cached_blocks` to find the hash for a block ID.
In production, maintain a reverse map `block_to_hash: HashMap<BlockId, BlockHash>`
for O(1) eviction. For this chapter's demo, the scan is acceptable.

### 7.2 Eviction Order

LRU: the block that was freed longest ago is evicted first. This is a
reasonable heuristic — recently freed blocks are more likely to be reused
by incoming requests with similar prefixes.

---

## 8. ref_count Semantics

Block ref_count tracks how many active sequences reference a physical block.

| Event | ref_count change | Consequence |
|-------|-----------------|-------------|
| `allocate()` | set to 1 | Block is in use by one sequence |
| Cache hit (`get_computed_blocks`) | +1 | Block shared by another sequence |
| Sequence finishes | -1 for each block | Block may still be used by others |
| ref_count reaches 0 | — | Block enters `free_queue`, stays in `cached_blocks` |
| `evict()` | — | Block removed from `cached_blocks` and `free_queue` |

### 8.1 Sharing Example

1. Request A allocates block 5 → ref_count = 1
2. Request B hits cache for block 5 → ref_count = 2
3. Request A finishes → ref_count = 1 (block still active for B)
4. Request B finishes → ref_count = 0 (block enters free_queue)
5. Request C hits cache for block 5 → ref_count = 1 (removed from free_queue)
6. Request C finishes → ref_count = 0 (back in free_queue)
7. New allocation needs a block → evict() removes block 5 from cache

---

## 9. Integration with BlockAllocator

The prefix cache sits between the scheduler and the raw block allocator:

```
function allocate_for_request(token_ids):
    // Step 1: Check prefix cache
    (cached_block_ids, num_computed) = prefix_cache.get_computed_blocks(token_ids)

    // Step 2: Allocate remaining blocks from free pool
    remaining_tokens = len(token_ids) - num_computed
    new_blocks_needed = ceil(remaining_tokens / block_size)
    new_block_ids = []
    for _ in 0..new_blocks_needed:
        if allocator.num_free_blocks() == 0:
            // Try evicting from prefix cache
            evicted = prefix_cache.evict()
            if evicted is None:
                return Error("out of memory")
            allocator.return_block(evicted)
        new_block_ids.append(allocator.allocate())

    // Step 3: Build block table from cached + new blocks
    block_table = BlockTable(cached_block_ids + new_block_ids, block_size)

    return (block_table, num_computed)
```

The `num_computed` value tells the engine how many tokens to skip during
prefill — those tokens already have valid KV data in the cached blocks.

---

## 10. Demo Program

The demo (`examples/ch16_prefix_caching`) uses mock data — no model needed.
It demonstrates prefix caching mechanics across 3 scenarios.

### 10.1 Setup

```
BLOCK_SIZE = 16
NUM_BLOCKS = 10  (160 total token slots)

Shared prefix: 64 tokens (4 full blocks)
  tokens = [100, 101, 102, ..., 163]  // 64 tokens, blocks 0-3
```

### 10.2 Scenario 1: First Request — Cache Miss

Request A: 80 tokens (shared 64-token prefix + 16 unique tokens)
- PrefixCache is empty
- `get_computed_blocks` → 0 hits, 4 lookups (all miss)
- Allocate 5 blocks, compute KV for all 80 tokens
- After prefill: `cache_full_blocks` inserts 4 full blocks (block 5 has
  only 16 tokens if there are 80, meaning 5 full blocks — all 5 cached)
- Show block hashes and cache state

### 10.3 Scenario 2: Second Request — Cache Hit

Request B: 80 tokens (same 64-token prefix + 16 different tokens)
- `get_computed_blocks` → 4 hits! Blocks for the shared prefix found
- Only need 1 new block for the unique suffix
- Prefill computation skipped for 64 tokens (4 blocks)
- Show: ref_count incremented on shared blocks

### 10.4 Scenario 3: Eviction and Reuse

- Request A finishes → decrement ref_count on its blocks
- Shared blocks still have ref_count=1 (Request B still active)
- Request B finishes → shared blocks drop to ref_count=0, enter free_queue
- Request C arrives with same prefix → cache hit even though blocks are "freed"
- Show: blocks reused from free_queue without eviction
- If pool is exhausted, show eviction of oldest free block

### 10.5 Expected Output Structure

```
=== PART 1: Hash Chaining ===
[Show 4 blocks of 16 tokens each]
[Show hash computation with chaining]
[Show that changing block 0 changes all subsequent hashes]

=== PART 2: Cache Miss — First Request ===
[Request A: 80 tokens, all blocks miss]
[5 blocks allocated, 0 cache hits]
[After prefill: 5 blocks cached]

=== PART 3: Cache Hit — Second Request ===
[Request B: same prefix, 4 blocks hit]
[1 new block allocated, 64 tokens skipped]
[ref_count on shared blocks: 2]

=== PART 4: Eviction — Freeing and Reusing ===
[Request A finishes, blocks freed]
[Request C reuses cached blocks]
[Show eviction when pool exhausted]

=== PART 5: Hit Rate Summary ===
[Total lookups, hits, hit rate percentage]

Chapter 16 complete. Next: Speculative Decoding (ch17)
```

---

## 11. Correctness Criteria

1. **Hash chaining:** Block N's hash depends on both its tokens AND block N-1's
   hash. Changing tokens in block 0 changes hashes of all subsequent blocks.
2. **Full blocks only:** Partial blocks (num_filled < block_size) are never
   hashed or cached.
3. **Contiguous prefix:** `get_computed_blocks` stops on the first miss —
   no gaps in the cached prefix.
4. **ref_count correct:** Cache hits increment ref_count. Sequence completion
   decrements it. Block enters free_queue only when ref_count reaches 0.
5. **Free but cached:** Blocks with ref_count=0 remain in `cached_blocks`
   and can serve future cache hits until evicted.
6. **LRU eviction:** `evict()` removes the oldest block from `free_queue`
   and deletes it from `cached_blocks`.
7. **No double-free:** A block in `free_queue` that gets a cache hit is
   removed from `free_queue` before its ref_count is incremented.
8. **Hit rate tracking:** Total lookups and hits are counted; hit rate =
   hits / lookups * 100.

---

## 12. Validation Summary

| Test | What it checks |
|------|---------------|
| Part 1 present | Hash chaining section exists |
| Part 2 present | Cache miss section exists |
| Part 3 present | Cache hit section exists |
| Part 4 present | Eviction section exists |
| Part 5 present | Hit rate section exists |
| Hash chaining | Block hash values shown, chaining mentioned |
| Cache miss | First request shows all misses, blocks allocated |
| Cache hit | Second request shows hits, prefill skipped |
| Eviction | LRU eviction mentioned, blocks freed and reused |
| Hit rate | Percentage value shown |
| Completion marker | "Chapter 16 complete" appears |
