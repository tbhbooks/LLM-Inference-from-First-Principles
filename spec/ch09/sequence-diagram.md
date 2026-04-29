# Chapter 9: Sequence Diagram

## Scenario 2: External Fragmentation Sequence

```mermaid
sequenceDiagram
    participant User
    participant Pool as MemoryPool<br/>(100 slots)

    User->>Pool: allocate A (20 slots)
    Note over Pool: [AAAA.................. ][..........................]
    Pool-->>User: OK (slots 0-19)

    User->>Pool: allocate B (15 slots)
    Note over Pool: [AAAAA..BBBBB.......... ][..........................]
    Pool-->>User: OK (slots 20-34)

    User->>Pool: allocate C (25 slots)
    Note over Pool: [AAAAA..BBBBB..CCCCCCCCC][CCC....................  ]
    Pool-->>User: OK (slots 35-59)

    User->>Pool: allocate D (15 slots)
    Note over Pool: [AAAAA..BBBBB..CCCCCCCCC][CCCDDDDD..............  ]
    Pool-->>User: OK (slots 60-74)

    User->>Pool: allocate E (10 slots)
    Note over Pool: [AAAAA..BBBBB..CCCCCCCCC][CCCDDDDDEEEE..........  ]
    Pool-->>User: OK (slots 75-84)

    User->>Pool: deallocate B
    Note over Pool: [AAAAA.........CCCCCCCCC][CCCDDDDDEEEE..........  ]
    Pool-->>User: freed 15 slots (20-34)

    User->>Pool: deallocate D
    Note over Pool: [AAAAA.........CCCCCCCCC][CCC.....EEEE..........  ]
    Pool-->>User: freed 15 slots (60-74)

    User->>Pool: allocate F (25 slots)?
    Note over Pool: 45 free slots<br/>but largest block = 15
    Pool-->>User: FAILED!
```
**Figure 9.4** — External fragmentation in action. After B and D depart, 45 slots are free but split into 3 fragments. Request F needs 25 contiguous slots — impossible.

## Contiguous vs Paged Allocation

```mermaid
sequenceDiagram
    participant Req as Request<br/>(10 tokens)
    participant Naive as Contiguous<br/>Allocator
    participant Paged as Block<br/>Allocator

    Note over Req,Paged: Same request, two allocation strategies

    Req->>Naive: need 10 tokens (max 30)
    Naive-->>Req: 30 contiguous slots<br/>20 wasted

    Req->>Paged: need 10 tokens
    Paged-->>Req: 1 block (16 slots)<br/>6 wasted
    Note over Paged: Block can be anywhere<br/>in memory — no fragmentation
```
**Figure 9.5** — Contiguous allocation reserves for the worst case (max_seq_len). Paged allocation only allocates what's needed, one block at a time.
