# Chapter 10: Interface Specification

## Overview

This chapter implements the core of PagedAttention: block-based KV cache management. Instead of allocating contiguous memory for each request's KV cache, we divide memory into fixed-size blocks and use a block table (analogous to an OS page table) to map each sequence's tokens to physical blocks.

This eliminates both internal and external fragmentation demonstrated in Chapter 9.

## Dependencies

- **Chapter 9**: Motivates the problem (fragmentation). No code dependency.
- **Chapters 3-8**: Existing module structure. The `memory/` module gets new files.

## New Data Types

### Block

A fixed-size unit of KV cache storage.

| Field | Type | Description |
|-------|------|-------------|
| `block_id` | BlockId (int) | Unique physical block identifier (0-based) |
| `block_size` | int | Number of token slots per block (e.g., 16) |
| `ref_count` | int | Number of sequences referencing this block |
| `num_filled` | int | Slots currently holding data (0..block_size) |

**Invariants:**
- `0 <= num_filled <= block_size`
- `ref_count >= 0`
- A block is "full" when `num_filled == block_size`
- A block is "free" when `ref_count == 0`

### BlockAllocator (trait/interface)

Manages a pool of physical blocks.

#### `allocate() -> BlockId`

Remove a block from the free pool and return its ID.

```
precondition: num_free_blocks() > 0
postcondition: num_free_blocks() decreases by 1
postcondition: returned block has ref_count = 1
```

Error if no free blocks available.

#### `free(block_id: BlockId)`

Decrement the block's ref_count. If it reaches 0, return block to free pool.

```
precondition: block exists and ref_count > 0
postcondition: ref_count decremented by 1
postcondition: if ref_count == 0, block is in free pool
```

#### `can_allocate(num_blocks: int) -> bool`

Check if at least `num_blocks` free blocks are available.

```
return num_free_blocks() >= num_blocks
```

#### `num_free_blocks() -> int`

Count of blocks currently in the free pool.

#### `num_total_blocks() -> int`

Total blocks managed (free + allocated).

### FreeListAllocator (implements BlockAllocator)

Simple implementation using a FIFO queue of free block IDs.

**Constructor: `FreeListAllocator(num_blocks: int, block_size: int)`**

```
blocks = [Block(id=i, block_size, ref_count=0, num_filled=0) for i in 0..num_blocks]
free_queue = deque([0, 1, 2, ..., num_blocks-1])
```

**allocate():**
```
block_id = free_queue.pop_front()    // FIFO: oldest free block first
blocks[block_id].ref_count = 1
blocks[block_id].num_filled = 0
return block_id
```

**free(block_id):**
```
blocks[block_id].ref_count -= 1
if blocks[block_id].ref_count == 0:
    blocks[block_id].num_filled = 0
    free_queue.push_back(block_id)
```

### BlockTable

Per-sequence mapping from logical blocks to physical blocks. This is the "page table" for one sequence.

| Field | Type | Description |
|-------|------|-------------|
| `block_ids` | list of BlockId | Physical block IDs in logical order |
| `block_size` | int | Tokens per block (same as allocator's block_size) |

#### `append_block(block_id: BlockId)`

Add a physical block to the end of this sequence's mapping.

```
block_ids.append(block_id)
```

#### `num_blocks() -> int`

```
return len(block_ids)
```

#### `num_tokens_capacity() -> int`

Total token slots available to this sequence.

```
return num_blocks() * block_size
```

#### `slot_for_token(token_position: int) -> (BlockId, int)`

Map a token position to its physical location.

```
block_index = token_position / block_size       // integer division
offset = token_position % block_size
block_id = block_ids[block_index]
return (block_id, offset)
```

**Precondition:** `token_position < num_tokens_capacity()`

#### `physical_slot(token_position: int) -> int`

Flat physical slot index for direct KV cache array addressing.

```
(block_id, offset) = slot_for_token(token_position)
return block_id * block_size + offset
```

**The slot mapping formula** (for reference):
```
token_position → block_index = token_position / block_size
              → offset = token_position % block_size
              → block_id = block_table.block_ids[block_index]
              → physical_slot = block_id * block_size + offset
```

This is identical to OS virtual-to-physical address translation:
```
virtual_address → page_number = virtual_address / page_size
               → offset = virtual_address % page_size
               → frame = page_table[page_number]
               → physical_address = frame * page_size + offset
```

## Helper Functions

### `blocks_needed(num_tokens: int, block_size: int) -> int`

How many blocks are needed to store `num_tokens` tokens.

```
return ceil(num_tokens / block_size)
```

### `waste_in_last_block(num_tokens: int, block_size: int) -> int`

Unused slots in the last (partially filled) block.

```
remainder = num_tokens % block_size
if remainder == 0: return 0
return block_size - remainder
```

## Constants (for demo program)

```
TOTAL_MEMORY_SLOTS = 160     // 10 blocks of 16
BLOCK_SIZE = 16
NUM_BLOCKS = TOTAL_MEMORY_SLOTS / BLOCK_SIZE = 10
```

## Demo Scenarios

### Scenario 1: Basic Allocation (1 request)

Request A: 37 tokens.
- Blocks needed: `ceil(37/16) = 3`
- Waste: `16 - (37 % 16) = 16 - 5 = 11` slots in last block
- Show: block IDs assigned, slot mapping for tokens 0, 16, 36

### Scenario 2: Multiple Requests

| Request | Tokens | Blocks | Waste |
|---------|--------|--------|-------|
| A | 10 | 1 | 6 |
| B | 25 | 2 | 7 |
| C | 50 | 4 | 14 |
| **Total** | **85** | **7/10** | **27** |

Compare with contiguous (max_seq_len=50): 50*3=150, only 2 fit in 160 slots. 1 rejected.
Paged: all 3 fit, 3 blocks free.

### Scenario 3: Arrival and Departure

Steps with block counts:
1. Allocate A (1 block) → 9 free
2. Allocate B (2 blocks) → 7 free
3. Allocate C (4 blocks) → 3 free
4. Free B (2 blocks returned) → 5 free
5. Allocate D (3 blocks) → 2 free. D's blocks may include B's old blocks.

Key insight: D's blocks are non-contiguous — and it does not matter.

### Scenario 4: Slot Mapping Deep Dive

For a request with 37 tokens across blocks [4, 1, 7]:

| Token | Block Index | Block ID | Offset | Physical Slot |
|-------|-------------|----------|--------|---------------|
| 0 | 0 | 4 | 0 | 64 |
| 8 | 0 | 4 | 8 | 72 |
| 16 | 1 | 1 | 0 | 16 |
| 24 | 1 | 1 | 8 | 24 |
| 32 | 2 | 7 | 0 | 112 |
| 36 | 2 | 7 | 4 | 116 |

Physical slots are not contiguous. The block table provides the mapping.

### Scenario 5: Head-to-Head Comparison

Pool: 160 slots (10 blocks). 5 requests with varying lengths.
max_seq_len for contiguous: 40.

| Request | Tokens | Contiguous (40 each) | Paged (blocks) |
|---------|--------|---------------------|----------------|
| A | 10 | 40 | 1 block (16) |
| B | 25 | 40 | 2 blocks (32) |
| C | 8 | 40 | 1 block (16) |
| D | 15 | 40 | 1 block (16) |
| E | 12 | REJECTED (160 full) | 1 block (16) |

Contiguous: 4 requests × 40 = 160 slots. 1 rejected.
Paged: 6 blocks × 16 = 96 slots used. All 5 served. 4 blocks free.

| Metric | Contiguous | Paged |
|--------|-----------|-------|
| Requests served | 4 | 5 |
| Slots used | 160 | 96 |
| Tokens stored | 58 | 70 |
| Waste % | 63.8% | 26.0% |
| Free slots | 0 | 64 |

## Output Sections

| Section | Title |
|---------|-------|
| PART 1 | Block Allocation — One Request at a Time |
| PART 2 | Multiple Requests — Sharing the Pool |
| PART 3 | Arrival and Departure — No Fragmentation |
| PART 4 | The Slot Mapping — Virtual to Physical |
| PART 5 | Head-to-Head — Contiguous vs Paged |
| PART 6 | Why This Changes Everything |

## Validation Rules

1. All 6 section headers present
2. "block" mentioned (the allocation unit)
3. Block IDs shown (e.g., "block 0", "block 1", or numeric IDs)
4. "slot" mentioned (physical slot mapping)
5. "16" present (block size)
6. Comparison showing paged serves more requests
7. "fragmentation" or "waste" terms used
8. "Chapter 10 complete" closing
