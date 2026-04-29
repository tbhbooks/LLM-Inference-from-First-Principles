# Chapter 16 -- LLM Prompt Template: Prefix Caching

Copy and paste the prompt below into your LLM of choice, or use it as a
reference while implementing by hand. This builds on the project from
Chapters 1-15.

---

## The Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 16.
I have an existing project from Chapters 1-15 with:
- Full GPT-2 model with KV cache (ch04-06)
- Greedy generation loop (ch07)
- PagedAttention memory management with BlockAllocator and BlockTable (ch09-10)
- Continuous batching scheduler (ch11-12)
- Engine loop with Sampler trait (ch13)
- Sampling strategies (ch14)
- API server (ch15)

Now implement prefix caching: content-addressed block reuse so that requests
sharing a common prompt prefix skip redundant KV computation.

LANGUAGE/FRAMEWORK: [Rust with candle / Python with PyTorch / your choice]

=== WHAT TO CREATE / MODIFY ===

  NEW FILES:
    src/memory/prefix_cache     <-- PrefixCache, hash_block, BlockHash
    examples/ch16_prefix_caching  <-- Demo program

  MODIFY:
    src/memory/block_allocator  <-- ref_count increment/decrement support
    src/memory/mod.rs           <-- Re-export PrefixCache, BlockHash

  KEEP UNCHANGED:
    src/model/ (all files)
    src/tokenizer/ (all files)
    src/sampling/ (all files)
    src/scheduler/ (all files)
    src/api/ (all files)

=== BLOCK HASH TYPE ===

  type BlockHash = uint64

  BlockHash is a content-based identifier for a full block of tokens.
  Two blocks with identical token content AND identical prefix history
  produce the same hash.

=== HASH CHAINING ===

  hash_block(tokens: list[TokenId], prev_hash: Option[BlockHash]) -> BlockHash

  The hash for block N incorporates:
  1. The tokens stored in block N
  2. The hash of block N-1 (the previous block in the sequence)

  This chaining prevents false cache hits: two blocks with the same tokens
  but different preceding context produce different hashes.

  Algorithm:
      state = hash_init()
      if prev_hash is not None:
          state = hash_update(state, prev_hash.to_bytes())
      for token in tokens:
          state = hash_update(state, token.to_bytes())
      return hash_finalize(state)

  IMPORTANT: Only full blocks (num_filled == block_size) are hashable.
  Partial blocks are not cached because their content may change.

=== PREFIX CACHE STRUCT ===

  struct PrefixCache:
      cached_blocks: HashMap<BlockHash, BlockId>
      free_queue: VecDeque<BlockId>    // LRU eviction queue
      block_size: int

  The cached_blocks map is the core lookup table. When a block is freed
  (ref_count drops to 0), it goes into free_queue but stays in cached_blocks.
  It can still serve cache hits until evicted.

=== GET COMPUTED BLOCKS ===

  get_computed_blocks(token_ids: list[TokenId]) -> (Vec<BlockId>, num_computed_tokens: int)

  Walk through the token sequence block by block:
  1. Split token_ids into full blocks of block_size tokens
  2. For each full block, compute its hash (chaining with the previous block's hash)
  3. Look up the hash in cached_blocks
  4. If found: add to result, increment ref_count on that block
  5. If NOT found: STOP — return what we have so far

  The key insight: we stop on the FIRST miss. Prefix caching only reuses
  contiguous prefix blocks. A hit at position 3 after a miss at position 2
  is not useful because the KV cache must be computed sequentially.

  Returns:
  - Vec<BlockId>: physical block IDs that can be reused (already have KV data)
  - num_computed_tokens: len(result) * block_size

=== CACHE FULL BLOCKS ===

  cache_full_blocks(block_ids: list[BlockId], token_ids: list[TokenId])

  After a request finishes prefill, insert its full blocks into the cache:
  1. For each full block (has block_size tokens):
     a. Compute the block hash (with chaining)
     b. Insert hash → block_id into cached_blocks
  2. Partial (last) block is NOT cached

=== EVICTION POLICY (LRU) ===

  evict() -> Option<BlockId>

  When the allocator needs a free block and the free pool is empty:
  1. Pop the oldest block from free_queue (front of deque)
  2. Remove its entry from cached_blocks (by looking up its hash)
  3. Return the block ID for reuse

  A block enters the free_queue when its ref_count drops to 0.
  It stays in cached_blocks (can still serve hits) until actually evicted.
  If a block in free_queue gets a cache hit before eviction, remove it
  from free_queue and increment its ref_count — it's back in active use.

=== REF_COUNT SEMANTICS ===

  Block ref_count rules:
  - allocate(): sets ref_count = 1
  - cache hit (get_computed_blocks finds it): ref_count += 1
  - request finishes: ref_count -= 1 for each block
  - ref_count drops to 0: block enters free_queue (but stays in cached_blocks)
  - evict(): removes from cached_blocks and free_queue, block is truly free

=== INTEGRATION WITH BLOCK ALLOCATOR ===

  The PrefixCache works alongside the BlockAllocator:
  - Before allocating new blocks, check PrefixCache for reusable prefix blocks
  - The allocator's free() method notifies PrefixCache to add blocks to free_queue
  - When allocator needs blocks and free pool is empty, ask PrefixCache to evict

=== DEMO PROGRAM (examples/ch16_prefix_caching) ===

The demo uses mock data (no model needed). It should show 3 scenarios
plus hash chaining mechanics and hit rate tracking.

Output format: 5 parts

PART 1: Hash Chaining
  - Show a 64-token prefix (4 blocks of 16 tokens each)
  - Compute hash for each block, showing how each hash chains to the previous
  - Show that changing one token in block 1 changes ALL subsequent hashes

PART 2: Cache Miss — First Request
  - Request A arrives with 80 tokens (5 blocks: 4 prefix + 1 unique)
  - PrefixCache is empty → all 5 blocks miss
  - Allocate 5 fresh blocks, compute KV for all 80 tokens
  - After prefill, cache the 4 full prefix blocks (block 5 is partial, not cached)
  - Show: 0 cache hits, 5 blocks allocated, 4 blocks cached

PART 3: Cache Hit — Second Request
  - Request B arrives with the SAME 64-token prefix + different 16 tokens
  - get_computed_blocks finds 4 cached blocks → cache hit!
  - Only need to allocate 1 new block and compute KV for 16 tokens (not 80)
  - Show: 4 cache hits, 1 new block allocated, prefill skipped for 64 tokens

PART 4: Eviction — Freeing and Reusing
  - Request A finishes → ref_count drops to 0 on its blocks
  - Blocks enter free_queue but remain in cached_blocks
  - Request C arrives with same prefix → cache hit! (blocks still in cache)
  - Request B also finishes → more blocks freed
  - Show eviction when a new request needs blocks and pool is exhausted
  - Show: blocks reused from cache, eviction of oldest unused blocks

PART 5: Hit Rate Summary
  - Show total queries, cache hits, and hit rate percentage
  - Example: "3 queries, 8 block hits out of 15 block lookups, 53.3% hit rate"

End with: "Chapter 16 complete. Next: Speculative Decoding (ch17)"

=== WHAT SUCCESS LOOKS LIKE ===

Running the demo produces clearly labeled output showing:
- How hash chaining prevents false cache hits
- That the first request with a new prefix always misses
- That subsequent requests with the same prefix get cache hits
- That freed blocks remain cacheable until evicted
- That eviction removes the oldest unused blocks
- A summary hit rate metric

Each section shows block IDs, hash values, and ref_count changes.
The output demonstrates significant prefill savings on repeated prefixes.

=== WHAT TO PRODUCE ===

Produce the complete set of NEW or MODIFIED source files.
Do NOT recreate files from earlier chapters that are unchanged.

After this chapter, the engine can skip redundant KV computation for
requests that share common prompt prefixes — a major throughput optimization
for chat and few-shot prompting workloads.
```
