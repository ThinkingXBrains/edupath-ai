# Model Strategy

## Qwen 3.8 27B

Used for:
- learner profiling
- learning-plan generation
- practice-task generation
- resource curation
- progress report
- learner Q&A

Why:
- current Groq model
- supports structured outputs
- supports reasoning
- lower-cost/lighter workload than GPT-OSS 120B
- 131K context window

## GPT-OSS 120B

Used for:
- deep assessment of learner submissions

Why:
- reserve the large model for the most reasoning-heavy stage
- keep 120B usage small enough to reduce rate-limit pressure

## Fallback

If the 120B assessment call is rate-limited, EduPath automatically retries the same assessment using Qwen 3.8 27B.

## Research

Web search is handled separately through DDGS, so research does not consume another LLM model.
