# Chapter 10: Sequence Diagram

## Block Allocation Lifecycle

```mermaid
sequenceDiagram
    participant Seq as Sequence
    participant BT as BlockTable
    participant Alloc as BlockAllocator
    participant Pool as Block Pool

    Note over Seq,Pool: Request arrives: 37 tokens needed

    Seq->>Alloc: can_allocate(3)?
    Alloc->>Pool: check free_queue.len >= 3
    Pool-->>Alloc: true (10 free)
    Alloc-->>Seq: yes

    loop 3 times (one per block needed)
        Seq->>Alloc: allocate()
        Alloc->>Pool: pop from free_queue
        Pool-->>Alloc: block_id (e.g., 0, 1, 2)
        Alloc-->>Seq: block_id
        Seq->>BT: append_block(block_id)
    end

    Note over BT: block_ids = [0, 1, 2]

    Note over Seq,Pool: During inference: map tokens to slots

    Seq->>BT: slot_for_token(37)
    BT-->>Seq: (block 2, offset 5)
    Seq->>BT: physical_slot(37)
    BT-->>Seq: 2 × 16 + 5 = 37

    Note over Seq,Pool: Request completes: free blocks

    loop 3 times
        Seq->>Alloc: free(block_id)
        Alloc->>Pool: decrement ref_count<br/>push to free_queue
    end

    Note over Pool: 10 free blocks again
```
**Figure 10.4** — Block allocation lifecycle. Allocate blocks on arrival, map tokens to physical slots during inference, free blocks on completion.

## Arrival and Departure — Block Reuse

```mermaid
sequenceDiagram
    participant A as Request A
    participant B as Request B
    participant D as Request D
    participant Alloc as Allocator<br/>(10 blocks)

    A->>Alloc: allocate 1 block
    Note over Alloc: 9 free
    B->>Alloc: allocate 2 blocks
    Note over Alloc: 7 free

    Note over A,Alloc: Time passes...

    B->>Alloc: free 2 blocks
    Note over Alloc: 9 free (B's blocks recycled)

    D->>Alloc: allocate 3 blocks
    Note over Alloc: 6 free
    Note over D: D may receive B's old blocks<br/>Non-contiguous? No problem!
```
**Figure 10.5** — Block reuse after deallocation. D's blocks may include B's recycled blocks. Unlike contiguous allocation, there is no fragmentation concern.
