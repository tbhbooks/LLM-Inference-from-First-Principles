# Chapter 18 -- Interface Specification: Structured Output

This is a language-agnostic specification. It defines the contracts for
grammar-constrained decoding using finite state machines — compile a pattern
to an FSM, precompute token validity per state, mask logits at each decode step.

---

## 1. Overview

Chapter 14 built the LogitsProcessor pipeline: temperature, top-k, top-p,
repetition penalty. Every processor transforms logits, but none of them
enforce *structure*. The model can still emit any token.

This chapter adds **grammar-constrained decoding**: a GrammarConstraint that
implements LogitsProcessor, uses a finite state machine (FSM) to track what
patterns are valid, and masks out tokens that would violate the constraint.

After this chapter, the engine can guarantee that generated output matches
a given pattern — numbers only, valid JSON keys, enum values, etc.

---

## 2. Dependencies

From Chapter 14: `LogitsProcessor` trait, `PipelineSampler`

The GrammarConstraint implements LogitsProcessor, so it plugs directly into
the existing sampling pipeline with no changes to the pipeline itself.

---

## 3. FSM Concepts

A **finite state machine** (FSM) is a graph of states connected by transitions.
Each transition is labeled with a character. The FSM starts in an initial state.
As it consumes characters, it follows transitions. If no transition exists for
a character, the input is rejected. If the FSM ends in an accept state, the
input is valid.

For structured output, the idea is:
1. Compile a pattern (e.g., regex `[0-9]+`) into an FSM.
2. At each decode step, check every token in the vocabulary: "if I appended
   this token's string to the output, would the FSM still be in a valid state?"
3. Mask out tokens that would leave the FSM in an invalid state (or with no
   transition at all).
4. Sample from the remaining valid tokens.

This guarantees the output matches the pattern.

---

## 4. FSMState Type

```
FSMState = usize    // unsigned integer identifying a state
```

States are numbered starting from 0. State 0 is always the initial state
by convention in this implementation.

---

## 5. SimpleFSM Struct

```
struct SimpleFSM:
    states: list[FSMState]                          // all states in the FSM
    initial_state: FSMState                         // starting state (always 0)
    accept_states: set[FSMState]                    // states where input is valid
    transitions: map[(FSMState, char), FSMState]    // state + char -> next state
```

### 5.1 advance Method

```
advance(state: FSMState, token_str: str) -> Option<FSMState>:
    // Walk the FSM through each character of the token string.
    // If any character has no transition, return None (token invalid).
    current = state
    for char in token_str:
        next = transitions.get((current, char))
        if next is None:
            return None    // no valid transition — token rejected
        current = next
    return Some(current)   // FSM landed in this state after consuming token
```

**Key insight:** A token is a *string* of one or more characters. The FSM must
consume the entire token string. If any character along the way has no
transition, the whole token is invalid from this state.

### 5.2 is_accepting Method

```
is_accepting(state: FSMState) -> bool:
    return state in accept_states
```

A generation is "complete" when the FSM is in an accept state and the model
emits an end-of-sequence token (or max tokens is reached).

### 5.3 Example: FSM for `[0-9]+`

| State | Accept? | Transitions |
|-------|---------|-------------|
| 0 | No | '0'-'9' -> State 1 |
| 1 | Yes | '0'-'9' -> State 1 |

- State 0: initial. Must see at least one digit to reach State 1.
- State 1: accept. Can see more digits (loops back to itself).
- Any non-digit character from any state -> no transition -> rejected.

### 5.4 Example: FSM for `[a-z]+`

| State | Accept? | Transitions |
|-------|---------|-------------|
| 0 | No | 'a'-'z' -> State 1 |
| 1 | Yes | 'a'-'z' -> State 1 |

Same structure — only the character class differs.

---

## 6. GrammarConstraint (LogitsProcessor Implementation)

```
struct GrammarConstraint:
    fsm: SimpleFSM
    current_state: FSMState                     // tracks FSM state across decode steps
    vocab: list[(TokenId, String)]              // vocabulary: token ID to string mapping
```

### 6.1 process Method (LogitsProcessor trait)

```
process(logits: FloatArray, token_ids_so_far: list[TokenId]) -> FloatArray:
    for (token_id, token_str) in vocab:
        next_state = fsm.advance(current_state, token_str)
        if next_state is None:
            logits[token_id] = -infinity    // mask invalid token
    return logits
```

**Behavior:**
- For each token in the vocabulary, check if the FSM can consume that token's
  string from the current state.
- If not, set that token's logit to negative infinity. After softmax, its
  probability becomes zero.
- Valid tokens keep their original logits. The model's preferences among
  valid tokens are preserved.
- This is applied *before* other processors (temperature, top-k, etc.) or
  after — the order relative to other processors is configurable, but
  grammar masking is typically done first (so other processors only see
  valid tokens).

### 6.2 accept_token Method (Not Part of LogitsProcessor)

```
accept_token(token_id: TokenId):
    token_str = vocab[token_id].string
    new_state = fsm.advance(current_state, token_str)
    assert new_state is not None    // token was validated during process()
    current_state = new_state
```

**This method is called by the engine after sampling**, not by the pipeline.
It advances the FSM state so the next call to `process()` checks validity
from the new state.

### 6.3 reset Method

```
reset():
    current_state = fsm.initial_state
```

Resets the constraint for a new generation.

---

## 7. Token Validity Check

For a given FSM state and vocabulary, we can classify every token:

```
function check_token_validity(
    fsm: SimpleFSM,
    state: FSMState,
    vocab: list[(TokenId, String)]
) -> list[(TokenId, String, bool)]:
    results = []
    for (token_id, token_str) in vocab:
        valid = fsm.advance(state, token_str) is not None
        results.append((token_id, token_str, valid))
    return results
```

**Example** — FSM for `[0-9]+`, State 0 (initial):

| Token ID | String | Valid? | Why |
|----------|--------|--------|-----|
| 0 | "4" | Yes | '4' is a digit, transition 0 -> 1 |
| 1 | "2" | Yes | '2' is a digit, transition 0 -> 1 |
| 2 | "42" | Yes | '4' -> state 1, '2' -> state 1 |
| 3 | "seven" | No | 's' has no transition from state 0 |
| 4 | "hello" | No | 'h' has no transition from state 0 |
| 5 | "0" | Yes | '0' is a digit |
| 6 | "9" | Yes | '9' is a digit |
| 7 | " " | No | space has no transition |
| 8 | "abc" | No | 'a' has no transition (digits only) |
| 9 | "forty" | No | 'f' has no transition |
| 10 | "two" | No | 't' has no transition |
| 11 | "." | No | '.' has no transition |
| 12 | "xyz" | No | 'x' has no transition |

Result: only tokens 0, 1, 2, 5, 6 are valid. All others get masked to
-infinity.

---

## 8. Bitmask Precomputation (Optional Optimization)

For production use, precompute a bitmask for each FSM state:

```
function precompute_masks(
    fsm: SimpleFSM,
    vocab: list[(TokenId, String)]
) -> map[FSMState, Bitmask]:
    masks = {}
    for state in fsm.states:
        mask = new Bitmask(len(vocab))    // all bits initially 0
        for (token_id, token_str) in vocab:
            if fsm.advance(state, token_str) is not None:
                mask.set(token_id)        // mark token as valid
        masks[state] = mask
    return masks
```

At decode time, applying the mask is O(vocab_size) — just AND the logits
with the precomputed mask. No per-token FSM walks needed.

This optimization is mentioned for completeness but not required for the
chapter demo. The per-token check in `process()` is sufficient.

---

## 9. StructuredOutputConfig

User-facing configuration for structured output:

```
struct StructuredOutputConfig:
    pattern: String    // regex pattern (e.g., "[0-9]+", "[a-z]+")
```

### 9.1 Usage

```
config = StructuredOutputConfig(pattern="[0-9]+")
fsm = compile_regex(config.pattern)
constraint = GrammarConstraint(fsm=fsm, current_state=0, vocab=vocabulary)
// Add constraint to the LogitsProcessor pipeline
```

---

## 10. Regex Compilation (Subset)

A minimal regex compiler that supports enough for the demo:

```
function compile_regex(pattern: str) -> SimpleFSM:
    // Supported syntax:
    //   - Literal characters: 'a', 'b', '1', etc.
    //   - Character classes: [0-9], [a-z], [A-Z]
    //   - Quantifiers: + (one or more), * (zero or more)
    //
    // NOT a full regex engine. No alternation (|), no groups,
    // no lookahead, no backreferences.

    // Parse the pattern into segments:
    //   "[0-9]+" -> CharClass('0'-'9', quantifier=OneOrMore)
    //   "[a-z]*" -> CharClass('a'-'z', quantifier=ZeroOrMore)
    //   "abc"    -> Literal('a'), Literal('b'), Literal('c')

    // Build FSM from segments:
    //   For each segment, add states and transitions.
    //   Quantifier '+' means: state_n -> state_n+1 on match,
    //                         state_n+1 -> state_n+1 on match (loop)
    //                         state_n+1 is accept if last segment
    //   Quantifier '*' means: state_n is accept (can match zero),
    //                         state_n -> state_n on match (loop)
```

### 10.1 Example Compilation: `[0-9]+`

```
Input:  "[0-9]+"
Parse:  CharClass(chars='0123456789', quantifier=OneOrMore)
Build:
  State 0 (initial, not accept):
    '0' -> 1, '1' -> 1, '2' -> 1, ..., '9' -> 1
  State 1 (accept):
    '0' -> 1, '1' -> 1, '2' -> 1, ..., '9' -> 1
Result: SimpleFSM(states=[0,1], initial=0, accept={1}, transitions={...})
```

### 10.2 Example Compilation: `[a-z]+`

```
Input:  "[a-z]+"
Parse:  CharClass(chars='abcdefghijklmnopqrstuvwxyz', quantifier=OneOrMore)
Build:
  State 0 (initial, not accept):
    'a' -> 1, 'b' -> 1, ..., 'z' -> 1
  State 1 (accept):
    'a' -> 1, 'b' -> 1, ..., 'z' -> 1
Result: SimpleFSM(states=[0,1], initial=0, accept={1}, transitions={...})
```

---

## 11. Integration with Sampling Pipeline

The GrammarConstraint plugs into the existing LogitsProcessor pipeline from
Chapter 14. No changes to PipelineSampler are needed.

```
// Build the processor list with grammar constraint first
processors = []
processors.append(grammar_constraint)           // mask invalid tokens first
processors.append(RepetitionPenaltyProcessor)   // then apply penalties
processors.append(TemperatureProcessor)          // scale
processors.append(TopKProcessor)                 // filter
processors.append(TopPProcessor)                 // nucleus filter

// PipelineSampler runs them in order as usual
sampler = PipelineSampler(processors, temperature, rng)
token_id = sampler.sample(logits, tokens_so_far)

// After sampling, advance the FSM
grammar_constraint.accept_token(token_id)
```

### 11.1 Pipeline Order with Grammar Constraint

| Order | Processor | Why this position |
|-------|-----------|-------------------|
| 1 | GrammarConstraint | Must run first — eliminates structurally invalid tokens before any other processing |
| 2 | RepetitionPenalty | Operates on remaining valid logits |
| 3 | Temperature | Scales remaining logits |
| 4 | TopK | Coarse filter on valid tokens only |
| 5 | TopP | Fine filter on valid tokens only |

---

## 12. Demo Program

The demo (`examples/ch18_structured_output`) should:

1. Build a mock vocabulary (13 tokens with known strings)
2. Create mock logits that favor word tokens over digit tokens
3. Show 5 parts:

### 12.1 Expected Output Structure

```
=== PART 1: The Problem ===
[Why unconstrained LLM output has no structure guarantee]
[Show mock logits favoring "forty", "two", "hello" — word tokens]
[Greedy/sampling picks word tokens: "forty two" not "42"]

=== PART 2: FSM Construction ===
[Compile [0-9]+ pattern to an FSM]
[Show states: State 0 (initial), State 1 (accept)]
[Show transitions: '0'-'9' from each state]
[Show which tokens are valid from State 0]

=== PART 3: Unconstrained vs Constrained ===
[Same logits for both]
[Unconstrained: all tokens eligible, picks "forty" (highest logit)]
[Constrained: only digit tokens survive mask, picks "4" or "42"]
[Clear before/after comparison]

=== PART 4: Token Masking ===
[Show all 13 tokens with valid/invalid status for State 0]
[Show logits before masking: all tokens have finite values]
[Show logits after masking: invalid tokens at -inf]
[Show that only 5 tokens survive (the digit tokens)]

=== PART 5: Pipeline Integration ===
[Build a PipelineSampler with GrammarConstraint + Temperature]
[Show the processor list]
[Show that GrammarConstraint is just another LogitsProcessor]
[Show a multi-step generation: sample token, advance FSM, repeat]

Chapter 18 complete. Next: Parallelism (ch19)
```

---

## 13. Correctness Criteria

1. **FSM advance is correct:** For `[0-9]+`, digit-only tokens return
   `Some(state)`, non-digit tokens return `None`.
2. **Masking works:** Invalid tokens get logits set to `-infinity`.
3. **Valid tokens preserved:** Tokens that pass the FSM check keep their
   original logits unchanged.
4. **FSM state advances:** After `accept_token()`, the current state
   updates correctly.
5. **Constrained output differs:** Same logits produce different output
   with vs. without the constraint.
6. **Pipeline integration:** GrammarConstraint implements LogitsProcessor
   and works inside PipelineSampler without modification.
7. **Accept states correct:** `[0-9]+` has State 1 as accept (must have
   at least one digit). State 0 is not accept.
8. **Multi-character tokens:** Tokens like "42" are checked by walking
   all characters through the FSM, not just the first character.

---

## 14. Validation Summary

| Test | What it checks |
|------|---------------|
| Part 1 present | "The Problem" section exists |
| Part 2 present | FSM construction section with states/transitions |
| Part 3 present | Unconstrained vs constrained comparison |
| Part 4 present | Token masking section with valid/invalid |
| Part 5 present | Pipeline integration section |
| FSM mentioned | Output mentions "fsm" or "state machine" or "finite state" |
| Constrained output | Output mentions "constrained" or "mask" and shows tokens filtered |
| Unconstrained shown | Output shows unconstrained generation for comparison |
| LogitsProcessor | Output mentions "logitsprocessor" or "pipeline" |
| Completion marker | "Chapter 18 complete" appears |
