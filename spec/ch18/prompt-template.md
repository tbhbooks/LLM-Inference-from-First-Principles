# Chapter 18 -- LLM Prompt Template: Structured Output

Copy and paste the prompt below into your LLM of choice, or use it as a
reference while implementing by hand. This builds on the project from
Chapters 1-17.

---

## The Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 18.
I have an existing project from Chapters 1-17 with:
- Full GPT-2 model with KV cache (ch04-06)
- Greedy generation loop (ch07)
- PagedAttention memory management (ch09-10)
- Continuous batching scheduler (ch11-12)
- Engine loop with Sampler trait (ch13)
- Sampling pipeline with LogitsProcessor chain (ch14)
- API server, streaming, multi-model support (ch15-17)

Now implement grammar-constrained decoding using finite state machines:
compile a pattern to an FSM, precompute token validity per state,
mask logits at each decode step so the model can only produce valid output.

LANGUAGE/FRAMEWORK: [Rust with candle / Python with PyTorch / your choice]

=== WHAT TO CREATE / MODIFY ===

  NEW FILES:
    src/sampling/structured.rs     <-- SimpleFSM, GrammarConstraint
    examples/ch18_structured_output <-- Demo program

  MODIFY:
    src/sampling/mod.rs     <-- Re-export structured module, integrate GrammarConstraint
    src/types.rs            <-- Add StructuredOutputConfig

  KEEP UNCHANGED:
    src/model/ (all files)
    src/tokenizer/ (all files)
    src/memory/ (all files)
    src/scheduler/ (all files)
    src/api/ (all files)

=== FSM STATE TYPE ===

  FSMState = usize (or equivalent unsigned integer)

  Transition function:
      (FSMState, char) -> Option<FSMState>

  If no transition exists for (state, char), the input is rejected.

=== SIMPLE FSM STRUCT ===

  struct SimpleFSM:
      states: list[FSMState]
      initial_state: FSMState
      accept_states: set[FSMState]
      transitions: map[(FSMState, char), FSMState]

  Methods:

      advance(state: FSMState, token_str: str) -> Option<FSMState>:
          // Walk the FSM through each character of the token string
          current = state
          for char in token_str:
              next = transitions.get((current, char))
              if next is None:
                  return None    // token is invalid from this state
              current = next
          return Some(current)

      is_accepting(state: FSMState) -> bool:
          return state in accept_states

=== GRAMMAR CONSTRAINT (implements LogitsProcessor) ===

  struct GrammarConstraint:
      fsm: SimpleFSM
      current_state: FSMState
      vocab: list[(TokenId, String)]    // vocabulary: token ID to string mapping

  Implements LogitsProcessor trait (from ch14):

      process(logits: FloatArray, token_ids_so_far: list[TokenId]) -> FloatArray:
          for (token_id, token_str) in vocab:
              next_state = fsm.advance(current_state, token_str)
              if next_state is None:
                  logits[token_id] = -infinity    // mask invalid token
          return logits

      accept_token(token_id: TokenId):
          // Call after sampling to advance the FSM
          token_str = vocab[token_id].string
          new_state = fsm.advance(current_state, token_str)
          assert new_state is not None    // token was validated before sampling
          current_state = new_state

  Note: accept_token is NOT part of LogitsProcessor — it's an additional
  method called by the engine after a token is sampled.

=== STRUCTURED OUTPUT CONFIG ===

  struct StructuredOutputConfig:
      pattern: String    // regex pattern to constrain output to

=== REGEX COMPILATION (subset) ===

  function compile_regex(pattern: str) -> SimpleFSM:
      // Simple regex compiler that supports:
      //   - Literal characters: a, b, 1, 2, etc.
      //   - Character classes: [0-9], [a-z], [A-Z]
      //   - Quantifiers: + (one or more), * (zero or more)
      //
      // NOT a full regex engine — just enough for the demo.
      //
      // Example: "[0-9]+" compiles to:
      //   State 0 (initial): on '0'-'9' -> State 1
      //   State 1 (accept):  on '0'-'9' -> State 1 (loop)
      //
      // Example: "[a-z]+" compiles to:
      //   State 0 (initial): on 'a'-'z' -> State 1
      //   State 1 (accept):  on 'a'-'z' -> State 1 (loop)

=== DEMO PROGRAM (examples/ch18_structured_output) ===

The demo uses mock logits and a small vocabulary (10-20 tokens).
No model needed. It should:

1. Build a small vocabulary:
   Token 0: "4"       (digit)
   Token 1: "2"       (digit)
   Token 2: "42"      (digit string)
   Token 3: "seven"   (word)
   Token 4: "hello"   (word)
   Token 5: "0"       (digit)
   Token 6: "9"       (digit)
   Token 7: " "       (space)
   Token 8: "abc"     (lowercase letters)
   Token 9: "forty"   (word)
   Token 10: "two"    (word)
   Token 11: "."      (punctuation)
   Token 12: "xyz"    (lowercase letters)

2. Create mock logits that favor word tokens (to make the contrast clear):
   "forty" has highest logit, "two" second, "42" lower, digits lower still

3. Show 4 scenarios:
   - PART 1: The Problem — why unconstrained generation gives wrong format
   - PART 2: FSM Construction — show states and transitions for [0-9]+
   - PART 3: Unconstrained vs Constrained — same logits, different masks,
     different output
   - PART 4: Token Masking — show which tokens valid/invalid per state,
     logits before/after masking
   - PART 5: Pipeline Integration — GrammarConstraint as LogitsProcessor
     in the PipelineSampler

Output format: 5 parts (The Problem, FSM Construction, Unconstrained vs
Constrained, Token Masking, Pipeline Integration).

End with: "Chapter 18 complete. Next: Parallelism (ch19)"

=== WHAT SUCCESS LOOKS LIKE ===

Running the demo produces clearly labeled output showing:
- An FSM with states and transitions for a digit pattern [0-9]+
- Unconstrained sampling picks word tokens (highest logits)
- Constrained sampling masks non-digit tokens to -infinity
- Only digit tokens survive the mask, producing valid numeric output
- The GrammarConstraint plugs into the existing LogitsProcessor pipeline

Each section shows logit values and which tokens are valid/invalid.
The output demonstrates that the same logits produce different outputs
depending on whether grammar constraints are applied.

=== WHAT TO PRODUCE ===

Produce the complete set of NEW or MODIFIED source files.
Do NOT recreate files from earlier chapters that are unchanged.

After this chapter, the engine can constrain generation output to match
a pattern (regex) using finite state machine guided decoding.
```
