# Chapter 9: Interface Specification

## Overview

This program simulates memory allocation for KV caches under naive contiguous allocation. It demonstrates internal fragmentation (wasted space within allocations) and external fragmentation (free space that can't be used because it's not contiguous). No model loading, no GPU — pure simulation with ASCII visualization.

The key insight: contiguous allocation is the bottleneck. The KV cache doesn't need contiguous memory — it needs a page table. This motivates PagedAttention in Chapter 10.

## Dependencies

- **Chapter 1**: Uses the same `section()` and separator formatting conventions.
- **Chapters 1-8**: The MVP is complete. This chapter adds a standalone example, no modifications to existing code.

## Data Types

### MemoryPool

A fixed-size array of token slots representing GPU memory available for KV cache.

| Field | Type | Description |
|-------|------|-------------|
| `total_slots` | int | Total number of token slots in the pool |
| `slots` | array of int | Each element is 0 (free) or a request_id (1-based) |

### Methods on MemoryPool

#### `allocate_contiguous(request_id: int, num_slots: int) -> (int, bool)`

Scans `slots` left-to-right for the first contiguous run of `num_slots` free (0) entries.

```
for start in 0..total_slots:
    if slots[start..start+num_slots] are all 0:
        mark all as request_id
        return (start, true)
return (0, false)
```

#### `deallocate(request_id: int)`

Sets all slots matching `request_id` back to 0.

```
for i in 0..total_slots:
    if slots[i] == request_id:
        slots[i] = 0
```

#### `utilization() -> float`

Fraction of slots that are non-zero.

```
occupied = count of slots[i] != 0
return occupied / total_slots
```

#### `internal_waste(allocated_slots: int, actual_tokens: int) -> int`

Wasted slots within a single allocation.

```
return allocated_slots - actual_tokens
```

#### `largest_free_block() -> int`

Length of the longest contiguous run of free (0) slots.

```
max_run = 0
current_run = 0
for each slot:
    if slot == 0: current_run += 1
    else: max_run = max(max_run, current_run); current_run = 0
return max(max_run, current_run)
```

#### `total_free() -> int`

Count of all free (0) slots.

#### `num_fragments() -> int`

Number of separate contiguous free regions.

```
fragments = 0
in_free = false
for each slot:
    if slot == 0 and not in_free:
        fragments += 1
        in_free = true
    elif slot != 0:
        in_free = false
return fragments
```

#### `visualize(width: int) -> string`

Render the pool as ASCII art. Each slot maps to one character:
- `0` → `.` (free)
- `1` → `A`, `2` → `B`, ..., `26` → `Z`

Break into rows of `width` characters. Prefix each row with its start index.

Example (total_slots=20, width=10):
```
  [00] AAAA......
  [10] BBBB......
```

### Request

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | int | Unique identifier (1-based, maps to 'A'=1, 'B'=2, etc.) |
| `prompt_len` | int | Actual number of tokens the request uses |
| `max_seq_len` | int | Allocated buffer size (worst-case reservation) |

## Constants

```
TOTAL_SLOTS = 100
DISPLAY_WIDTH = 50
```

## Simulation Scenarios

### Scenario 1: Internal Fragmentation

Three requests, all allocated max_seq_len=30 but using different amounts:

| Request | ID | prompt_len | max_seq_len | waste |
|---------|---|----|----|----|
| A | 1 | 10 | 30 | 20 |
| B | 2 | 25 | 30 | 5 |
| C | 3 | 8 | 30 | 22 |

**Totals:** 90 slots allocated, 43 tokens used, 47 wasted.
**Internal waste: 47/90 = 52.2%**

Steps: allocate A, B, C in order. After each, show visualization and waste.

### Scenario 2: External Fragmentation

Five requests allocated, then selective deallocation creates gaps.

| Step | Action | Request | max_seq_len | Slots used after |
|------|--------|---------|-------------|-----------------|
| 1 | Allocate A | 1 | 20 | 0-19 |
| 2 | Allocate B | 2 | 15 | 20-34 |
| 3 | Allocate C | 3 | 25 | 35-59 |
| 4 | Allocate D | 4 | 15 | 60-74 |
| 5 | Allocate E | 5 | 10 | 75-84 |
| 6 | Deallocate B | 2 | — | frees 20-34 |
| 7 | Deallocate D | 4 | — | frees 60-74 |
| 8 | Allocate F | 6 | 25 | **FAILS** |

After step 7: 45 free slots total, but largest contiguous block = 15.
Request F needs 25 contiguous slots — impossible despite 45 free.

### Scenario 3: The Scaling Wall

max_seq_len=30 for all requests. Allocate until failure.

```
floor(100 / 30) = 3 requests fit
remaining = 100 - (3 * 30) = 10 slots wasted (too small for another request)
waste = 10%
```

### Scenario 4: The OS Analogy

Print comparison table (see prompt-template for exact content).

## Output Sections

| Section | Title |
|---------|-------|
| PART 1 | Internal Fragmentation — The Waste Problem |
| PART 2 | External Fragmentation — The Tetris Problem |
| PART 3 | The Scaling Wall |
| PART 4 | From RAM to VRAM — The OS Analogy |
| PART 5 | What Paging Would Fix |
| PART 6 | The Key Insight |

## Validation Rules

The test suite checks for:

1. All 6 section headers present ("PART 1" through "PART 6")
2. "Internal Fragmentation" and "External Fragmentation" in titles
3. "52.2%" internal waste percentage
4. Allocation failure indicator ("FAILED" or "FAIL")
5. Fragmentation gap: "45" free slots, largest block "15"
6. Scaling wall: "3 requests" or "3" fitting
7. OS analogy terms: "page", "block", "page table" or "block table"
8. ASCII memory visualization with '.' characters
9. "Chapter 9 complete" closing
