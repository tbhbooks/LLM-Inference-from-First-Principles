# Chapter 9 -- LLM Prompt Template

Copy and paste this prompt into your LLM of choice to generate a working
implementation. This chapter adds a memory fragmentation simulator that
demonstrates why naive KV cache allocation wastes GPU memory.

---

## Prompt

```
I am building an LLM inference engine called "rvllm" as a learning project.
This is Chapter 9. I have a working MVP from Chapters 1-8: model loading,
forward pass, KV cache, greedy generation, and polished CLI.

Now I need to build a MEMORY FRAGMENTATION SIMULATOR that shows why naive
contiguous KV cache allocation breaks down under concurrent requests.
This is a standalone example — no model loading, no GPU. Pure simulation
with ASCII visualization.

TARGET LANGUAGE: [Rust / Python / Go / your choice]

=== WHAT TO CREATE ===

NEW FILES:
  examples/ch09_memory_problem.[ext]    <-- the program for this chapter

KEEP UNCHANGED:
  Everything from chapters 1-8.

=== OVERVIEW ===

This program simulates what happens when multiple requests with different
sequence lengths compete for a fixed GPU memory pool. It demonstrates:

1. CONTIGUOUS allocation (naive): each request gets a single contiguous buffer
   sized for max_seq_len, even if the actual sequence is shorter
2. The fragmentation problem: internal waste (allocated but unused) and
   external waste (gaps too small to use)
3. Why this motivates block-based allocation (Chapter 10)

=== DATA STRUCTURES ===

== MemoryPool ==

Represents a fixed region of GPU memory as an array of slots.

Fields:
    total_slots: int         — total number of token slots
    slots: array of int      — each slot is 0 (free) or request_id (occupied)

Methods:
    allocate_contiguous(request_id, num_slots) -> (start_index, bool)
        Scan for the first contiguous run of `num_slots` free slots.
        If found: mark all as request_id, return (start_index, true).
        If not found: return (0, false).

    deallocate(request_id)
        Free all slots belonging to request_id (set to 0).

    utilization() -> float
        Fraction of slots that are occupied (0.0 to 1.0).

    internal_waste(request_id, actual_tokens_used) -> int
        For a request allocated max_seq_len slots but only using
        actual_tokens_used: return (max_seq_len - actual_tokens_used).

    largest_free_block() -> int
        Scan the slots array and return the length of the longest
        contiguous run of free slots.

    total_free() -> int
        Count of all free (0) slots.

    num_fragments() -> int
        Count the number of separate contiguous free regions.
        E.g., [0,0,1,1,0,0,0,1,0] has 3 fragments: [0,0], [0,0,0], [0].

    visualize(width) -> string
        Render the memory pool as ASCII art. Each slot maps to one character:
        '.' for free, 'A'-'Z' for request 1-26 (request_id maps to letter).
        Break into rows of `width` characters.

== Request ==

    request_id: int
    prompt_len: int          — actual number of tokens the request uses
    max_seq_len: int         — allocated buffer size (worst-case)

== SimulationResult ==

    step: int
    action: string           — "ALLOCATE" or "DEALLOCATE"
    request_id: int
    success: bool
    utilization: float
    internal_waste: int
    total_free: int
    largest_free_block: int
    num_fragments: int

=== SIMULATION SCENARIOS ===

Use these constants:
    TOTAL_SLOTS = 100
    DISPLAY_WIDTH = 50

== Scenario 1: The Waste Problem (Internal Fragmentation) ==

Three requests arrive. All are allocated max_seq_len = 30 slots,
but they use different amounts:

    Request A (id=1): prompt_len=10, max_seq_len=30
    Request B (id=2): prompt_len=25, max_seq_len=30
    Request C (id=3): prompt_len=8,  max_seq_len=30

Steps:
1. Allocate A (30 slots) — show memory, report waste
2. Allocate B (30 slots) — show memory, report waste
3. Allocate C (30 slots) — show memory, report waste
4. Show summary: 90 slots allocated, only 43 tokens actually used,
   47 slots wasted (52.2% internal waste)

== Scenario 2: The Tetris Problem (External Fragmentation) ==

Start fresh. Requests arrive and depart, creating gaps.

    Request A (id=1): max_seq_len=20
    Request B (id=2): max_seq_len=15
    Request C (id=3): max_seq_len=25
    Request D (id=4): max_seq_len=15
    Request E (id=5): max_seq_len=10

Steps:
1. Allocate A (20 slots)
2. Allocate B (15 slots)
3. Allocate C (25 slots)
4. Allocate D (15 slots)
5. Allocate E (10 slots) — fills remaining 15 slots? No, only 10.
   Show: 85/100 used.
6. Deallocate B (free 15 slots, positions 20-34)
7. Deallocate D (free 15 slots, positions 60-74)
8. Now: 45 free slots total. Try to allocate Request F (id=6, max_seq_len=25).
   FAILS — largest contiguous block is only 15! 45 free slots but can't use them.
9. Show the gap: "45 free slots, but largest contiguous block = 15"

== Scenario 3: The Scaling Wall ==

Show how many concurrent requests fit as request count grows.
max_seq_len = 30 for all requests.

Allocate requests one at a time until allocation fails.
Report: how many fit, how much is wasted.

Expected: floor(100/30) = 3 requests fit. 10 slots wasted (can't fit a 4th).

== Scenario 4: The OS Analogy ==

Print a comparison table:

    Concept               OS Virtual Memory        LLM KV Cache (Naive)      LLM KV Cache (Paged)
    --------------------------------------------------------------------------------------------
    Allocation unit       Page (4 KB)              Contiguous buffer          Block (e.g., 16 tokens)
    Fragmentation         Solved by paging         Internal + external        Solved by paging
    Waste                 < 1 page per alloc       Up to max_seq_len - 1     < 1 block per alloc
    Address mapping       Page table               None (direct)             Block table
    Concurrent users      Thousands                Single digits             Hundreds

=== OUTPUT FORMAT ===

The program prints 6 sections. Each section header uses the same
section() format from ch01:
- blank line
- 78 '=' characters
- "  {TITLE}" (2-space indent)
- 78 '=' characters
- blank line

Sections:

PART 1: Internal Fragmentation — The Waste Problem
PART 2: External Fragmentation — The Tetris Problem
PART 3: The Scaling Wall
PART 4: From RAM to VRAM — The OS Analogy
PART 5: What Paging Would Fix
PART 6: The Key Insight

PART 5 prints a summary of what block-based allocation (paging) would solve:
- Internal waste: reduced from ~50% to < 1 block per request
- External fragmentation: eliminated (blocks need not be contiguous)
- Utilization: near 100% instead of ~50-60%
- Concurrent requests: 3-10x more with the same memory

PART 6 prints the key insight in an ASCII box:
  "The KV cache doesn't need contiguous memory. It needs a page table."

Closing: separator with
  "  Chapter 9 complete. Next: PagedAttention (ch10)"

=== VALIDATION ===

Your output should contain these strings:
- "PART 1" through "PART 6"
- "Internal Fragmentation" and "External Fragmentation"
- "52.2%" (internal waste percentage in scenario 1)
- "FAILED" or "FAIL" (allocation failure in scenario 2)
- "45 free" and "15" (the fragmentation gap in scenario 2)
- "3 requests" (scaling wall result)
- "Page" or "page" (OS analogy)
- "Block" or "block" (paged allocation)
- "page table" or "block table" (the key insight)
- ASCII memory visualization with '.' for free slots
- "Chapter 9 complete"

=== WHAT TO PRODUCE ===

1. The complete source file for examples/ch09_memory_problem.[ext]
2. The program should be runnable standalone with no model dependencies

After this chapter, the project gains:
  examples/ch09_memory_problem.[ext]
```
