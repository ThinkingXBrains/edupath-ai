# EduPath Model Routing

## Current problem

On the hackathon/free account, Groq can enforce a very small rolling output-token budget. A large response can therefore fail even when the overall daily quota is not exhausted.

## Solution

The app uses:
- compact schemas
- compact prompts
- explicit `max_output_tokens`
- client-side rolling output reservations
- two-stage 120B assessment
- no LLM call for progress
- no LLM call for web search

## Routing

Qwen 3.8 27B:
profile, plan, practice, Q&A.

GPT-OSS 120B:
assessment notes + final assessment.

Research:
DDGS web search.

Python:
mastery update, gap calculation, progress report, session state.

Splitting a prompt only helps when BOTH calls are individually and jointly small enough for the rolling output limit.
