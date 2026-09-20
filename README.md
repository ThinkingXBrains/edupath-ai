---
title: EduPath
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
---

# 🎓 EduPath — Token-Budgeted Hybrid Learning Agent

EduPath personalizes learning using two Groq-hosted LLMs plus deterministic Python state logic.

## Two-model strategy

### Qwen 3.8 27B
Used for:
- learner profile
- learning plan
- practice task
- learner Q&A

Every prompt and output is deliberately compact. ADK `GenerateContentConfig.max_output_tokens` is used to cap output.

### GPT-OSS 120B
Used only for deep practice assessment.

The assessment is split into two compact stages:

1. **Deep notes** — small evidence analysis.
2. **Final assessment** — converts the notes into the final structured schema.

If Stage 2 is rate-limited, the compact notes become a valid deterministic fallback assessment.

## Token protection

A conservative rolling output-token reservation is maintained per model so rapid button clicks do not deliberately burst the free-tier output budget.

## Research

Research uses the `ddgs` web-search package directly and does not consume an LLM generation call.

## Progress

Progress reporting is deterministic from the learner state and assessment evidence; no extra LLM call is needed.

## Render

Build:

```text
pip install -r requirements.txt
```

Start:

```text
python app.py
```

Required environment variable:

```text
GROQ_API_KEY
```

Optional model overrides:

```text
FAST_MODEL_NAME=groq/qwen/qwen3.8-27b
DEEP_MODEL_NAME=groq/openai/gpt-oss-120b
```

Never commit your Groq API key.
