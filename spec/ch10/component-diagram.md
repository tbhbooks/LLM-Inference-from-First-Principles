# Chapter 10: Component Diagram

## Block-Based KV Cache — Class Structure

```mermaid
classDiagram
    class BlockAllocator {
        <<trait>>
        +allocate() BlockId
        +free(block_id: BlockId)
        +can_allocate(num_blocks: int) bool
        +num_free_blocks() int
        +num_total_blocks() int
    }

    class FreeListAllocator {
        -blocks: Vec~Block~
        -free_queue: Deque~BlockId~
        -block_size: int
        +allocate() BlockId
        +free(block_id: BlockId)
        +can_allocate(num_blocks: int) bool
        +num_free_blocks() int
        +num_total_blocks() int
    }

    class Block {
        +BlockId block_id
        +int block_size
        +int ref_count
        +int num_filled
        +is_full() bool
        +is_free() bool
    }

    class BlockTable {
        +Vec~BlockId~ block_ids
        +int block_size
        +append_block(block_id: BlockId)
        +num_blocks() int
        +num_tokens_capacity() int
        +slot_for_token(pos: int) (BlockId, int)
        +physical_slot(pos: int) int
    }

    BlockAllocator <|.. FreeListAllocator : implements
    FreeListAllocator --> Block : manages pool of
    BlockTable --> Block : references by ID
```
**Figure 10.1** — PagedAttention class structure. The allocator manages a pool of blocks; each sequence holds a block table mapping logical positions to physical blocks.

## Slot Mapping: Virtual to Physical

```mermaid
flowchart LR
    T["Token position 37"] --> BI["block_index = 37 / 16 = 2"]
    BI --> OFF["offset = 37 % 16 = 5"]
    BI --> BT["block_table[2] = block 7"]
    BT --> PS["physical_slot = 7 × 16 + 5 = 117"]
    OFF --> PS

```
**Figure 10.2** — Slot mapping formula. A token position is split into block index and offset, then the block table maps logical block to physical block ID.

## Contiguous vs Paged Allocation

```mermaid
graph TB
    subgraph "Contiguous (Chapter 8)"
        CA["Request A<br/>10 tokens"] --> CB["40 contiguous slots<br/>30 wasted"]
        CC["Request B<br/>25 tokens"] --> CD["40 contiguous slots<br/>15 wasted"]
        CE["Request C<br/>8 tokens"] --> CF["40 contiguous slots<br/>32 wasted"]
        CG["Request D<br/>15 tokens"] --> CH["REJECTED<br/>no space left"]
    end

    subgraph "Paged (Chapter 10)"
        PA["Request A<br/>10 tokens"] --> PB["1 block (16 slots)<br/>6 wasted"]
        PC["Request B<br/>25 tokens"] --> PD["2 blocks (32 slots)<br/>7 wasted"]
        PE["Request C<br/>8 tokens"] --> PF["1 block (16 slots)<br/>8 wasted"]
        PG["Request D<br/>15 tokens"] --> PH["1 block (16 slots)<br/>1 wasted"]
    end
```
**Figure 10.3** — Same four requests, two allocation strategies. Contiguous rejects D; paged serves all four with room to spare.
