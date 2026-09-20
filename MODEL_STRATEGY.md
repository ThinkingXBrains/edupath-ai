# EduPath — Final Assessment Architecture

## Root cause of the previous failure

The 120B assessment agent was asked to return a Pydantic/JSON-schema document
with a tight completion cap. Groq rejected the generation when the response
did not satisfy the structured schema.

## Final fix

Assessment deliberately does NOT use structured JSON output.

GPT-OSS 120B is still used twice:

1. **Stage 1 — deep compact analysis**
   - task + submission
   - six short text lines
   - max 120 completion tokens

2. **Stage 2 — compact review**
   - Stage 1 notes only
   - five short text lines
   - max 110 completion tokens

Python parses these lines into `AssessmentResult`.

If Stage 2 fails, Stage 1 is already a complete assessment and is used directly.
No third LLM call is made.

## Deterministic responsibilities

Python owns:
- skill normalization
- mastery calculation
- gap recalculation
- final assessment schema construction
- fallback behavior

This removes JSON-schema validation from the fragile deep-assessment path.
