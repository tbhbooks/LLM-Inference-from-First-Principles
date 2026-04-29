# Chapter 10 -- LLM Prompt Template

Copy and paste this prompt into your LLM of choice to generate a working
implementation. This chapter implements PagedAttention — block-based KV
cache management that eliminates the fragmentation shown in Chapter 9.

---

## Prompt

```
I am building an LLM inference engine called "rvllm" as a learning project.
This is Chapter 10. I have a working MVP from Chapters 1-8 (model loading,
forward pass, KV cache, greedy generation) and a memory fragmentation
simulator from Chapter 9.

Now I need to implement PAGEDATTENTION — block-based KV cache management.
This replaces the naive contiguous KV cache with a paged system inspired
by OS virtual memory.

This chapter builds a standalone PagedAttention simulator/demo that shows
block allocation, block tables, slot mapping, and utilization — then
compares it against the contiguous approach from Chapter 9.

TARGET LANGUAGE: [Rust / Python / Go / your choice]

=== WHAT TO CREATE ===

NEW FILES:
  examples/ch10_paged_attention.[ext]    <-- the program for this chapter
  src/memory/block_allocator.[ext]       <-- BlockAllocator trait + FreeListAllocator
  src/memory/block_table.[ext]           <-- BlockTable: per-sequence block mapping

MODIFY:
  src/memory/mod.[ext]                   <-- export new types

KEEP UNCHANGED:
  Everything from chapters 1-8.

=== CORE CONCEPTS ===

== Block ==

A fixed-size unit of KV cache storage.

Fields:
    block_id: int            — unique physical block identifier
    block_size: int          — number of token slots per block (e.g., 16)
    ref_count: int           — how many sequences reference this block (for copy-on-write)
    num_filled: int          — how many slots currently hold data (0..block_size)

A block is "full" when num_filled == block_size.
A block is "free" when ref_count == 0.

== BlockAllocator (trait/interface) ==

Manages a pool of physical blocks.

Methods:
    allocate() -> block_id
        Pop a free block from the pool. Error if none available.

    free(block_id)
        Decrement ref_count. If ref_count reaches 0, return block to free pool.

    can_allocate(num_blocks) -> bool
        Check if at least num_blocks free blocks are available.

    num_free_blocks() -> int
        Count of blocks in the free pool.

    num_total_blocks() -> int
        Total blocks managed by this allocator.

== FreeListAllocator (implements BlockAllocator) ==

Simple implementation using a queue/deque of free block IDs.

Constructor:
    FreeListAllocator(num_blocks, block_size)
    Initialize all blocks with ref_count=0, add all to free_queue.

allocate():
    Pop from free_queue. Set ref_count=1. Return block_id.
    Error if free_queue is empty.

free(block_id):
    Decrement ref_count.
    If ref_count == 0: push back to free_queue.

== BlockTable ==

Per-sequence mapping from logical blocks to physical blocks.

Fields:
    block_ids: list of int     — physical block IDs in logical order
    block_size: int

Methods:
    append_block(block_id)
        Add a physical block to the end of this sequence's mapping.

    num_blocks() -> int
        Number of blocks allocated to this sequence.

    num_tokens_capacity() -> int
        Total token slots: num_blocks() * block_size.

    slot_for_token(token_position) -> (block_id, offset)
        Given an absolute token position, return which physical block
        and what offset within that block.

        block_index = token_position / block_size    (integer division)
        offset = token_position % block_size
        block_id = block_ids[block_index]
        return (block_id, offset)

    physical_slot(token_position) -> int
        Flat physical slot index for direct KV cache addressing.
        (block_id, offset) = slot_for_token(token_position)
        return block_id * block_size + offset

== Slot Mapping ==

The key insight: token positions map to physical slots through the block table.

    Token position 37 with block_size=16:
        block_index = 37 / 16 = 2   (third logical block)
        offset = 37 % 16 = 5        (sixth slot within that block)
        block_id = block_table[2]    (say, physical block 7)
        physical_slot = 7 * 16 + 5 = 117

    This is exactly how a page table works:
        virtual_address → (page_number, offset)
        page_number → physical_frame via page table
        physical_address = frame * page_size + offset

=== THE DEMO PROGRAM ===

The demo runs 5 scenarios comparing contiguous vs paged allocation.

Constants:
    TOTAL_MEMORY_SLOTS = 160   (fits 10 blocks of 16)
    BLOCK_SIZE = 16

== Scenario 1: Basic Allocation ==

Show block-by-block allocation for a single request.

    Request A: 37 tokens
    Contiguous: allocate 37 slots (or max_seq_len, whichever is larger)
    Paged: allocate ceil(37/16) = 3 blocks → 48 slots, waste = 11

    Print: how many blocks, which block IDs, slot mapping for token 0, 16, 37.

== Scenario 2: Multiple Requests ==

Three requests with different token counts:

    Request A: 10 tokens → 1 block
    Request B: 25 tokens → 2 blocks
    Request C: 50 tokens → 4 blocks

    Total blocks used: 7 out of 10.
    Waste: only in last block of each request.
    Equivalent contiguous waste: compare with max_seq_len=50.

    Print: block allocation per request, total utilization, waste comparison.

== Scenario 3: Arrival and Departure ==

Show that deallocation returns blocks to the pool cleanly.

    1. Allocate A (10 tokens, 1 block)
    2. Allocate B (25 tokens, 2 blocks)
    3. Allocate C (50 tokens, 4 blocks)       → 7/10 blocks used
    4. Deallocate B (free 2 blocks)            → 5/10 blocks used
    5. Allocate D (40 tokens, 3 blocks)        → 8/10 blocks used
    6. Show: D's blocks may be non-contiguous (reusing B's old blocks)

    Key point: no fragmentation! D gets blocks wherever they are free.

== Scenario 4: Slot Mapping Deep Dive ==

For Request A with 37 tokens allocated across 3 blocks (say blocks 4, 1, 7):

    Show the mapping for every 8th token:
    Token  0 → block 4, offset  0 → physical slot 64
    Token  8 → block 4, offset  8 → physical slot 72
    Token 16 → block 1, offset  0 → physical slot 16
    Token 24 → block 1, offset  8 → physical slot 24
    Token 32 → block 7, offset  0 → physical slot 112
    Token 36 → block 7, offset  4 → physical slot 116

    The physical slots are NOT contiguous — and that is the whole point.
    The attention kernel uses the block table to find each token's KV data.

== Scenario 5: Head-to-Head Comparison ==

Compare contiguous vs paged for the Chapter 9 workload:

    Pool: 160 slots
    Requests: varying lengths [10, 25, 8, 40, 15, 30, 12, 20]
    max_seq_len for contiguous: 50

    Contiguous: allocate 50 per request. Fits 3 requests (150/160).
                5 requests rejected.
    Paged:      allocate blocks as needed. All 8 requests fit.
                Total blocks: ceil(10/16)+ceil(25/16)+ceil(8/16)+...
                            = 1+2+1+3+1+2+1+2 = 13 blocks (out of 10)
                ... actually 13 > 10, so paged also runs out!
                Let's adjust: use pool of 320 slots (20 blocks).
                Contiguous: 50*8=400 > 320, fits 6.
                Paged: 13 blocks needed, 20 available. All 8 fit.

    (Adjust numbers so the comparison is dramatic.)

    Actually, use:
    Pool: 160 slots (10 blocks of 16)
    Requests: [10, 25, 8, 15, 12]  (5 requests)
    max_seq_len for contiguous: 40

    Contiguous: 40 per request. 4 fit (160/40=4). 1 rejected.
    Paged: ceil(10/16)+ceil(25/16)+ceil(8/16)+ceil(15/16)+ceil(12/16)
         = 1+2+1+1+1 = 6 blocks. All 5 fit. 4 blocks free.

    Print comparison table:
        Method      Requests Served    Slots Used    Waste %    Free Slots
        Contiguous  4                  160           60.0%      0
        Paged       5                  96            25.0%      64

=== OUTPUT FORMAT ===

6 sections using the standard section() format (78 '=' chars):

PART 1: Block Allocation — One Request at a Time
PART 2: Multiple Requests — Sharing the Pool
PART 3: Arrival and Departure — No Fragmentation
PART 4: The Slot Mapping — Virtual to Physical
PART 5: Head-to-Head — Contiguous vs Paged
PART 6: Why This Changes Everything

PART 6 summarizes the wins:
- Near-zero internal waste (only last block)
- Zero external fragmentation (blocks are interchangeable)
- Dynamic growth (allocate blocks as needed, not upfront)
- Copy-on-write possible (share blocks between requests)

Closing: "Chapter 10 complete. Next: Continuous Batching (ch11)"

=== VALIDATION ===

Your output should contain:
- "PART 1" through "PART 6"
- "block" and "Block" (the allocation unit)
- Block IDs (e.g., "block 0", "block 1")
- "slot" (physical slot mapping)
- "16" (block size)
- Comparison showing paged serves more requests than contiguous
- "fragmentation" or "waste" (comparison terms)
- "Chapter 10 complete"

=== WHAT TO PRODUCE ===

1. src/memory/block_allocator.[ext] — BlockAllocator trait + FreeListAllocator
2. src/memory/block_table.[ext] — BlockTable with slot mapping
3. src/memory/mod.[ext] — updated exports
4. examples/ch10_paged_attention.[ext] — the demo program

After this chapter:
  src/memory/
    mod.[ext]
    block_allocator.[ext]   (NEW)
    block_table.[ext]       (NEW)
  examples/
    ch10_paged_attention.[ext]  (NEW)
```
