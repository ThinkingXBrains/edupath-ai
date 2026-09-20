---
title: EduPath
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
---

# 🎓 EduPath — Adaptive Learning Companion

EduPath is an agentic AI learning and skill-gap application.

## Hybrid two-LLM architecture

EduPath deliberately routes work between two Groq-hosted models:

- **Qwen 3.8 27B** — default model for learner profiling, planning, practice generation, resource curation, progress reports and learner Q&A.
- **GPT-OSS 120B** — reserved for the deepest step: learner-work assessment. If the 120B call is rate-limited, the assessment falls back to Qwen 3.8 27B.

This reduces repeated use of the 120B rate-limit bucket while still demonstrating two-model routing.

## Research

The app uses the `ddgs` Python package for web search, then the Qwen 3.8 27B Resource Curator converts the search evidence into structured learning resources. No third LLM is required.

## Core flow

Profile → Skill Gaps → Plan → Practice → Assessment → Mastery Update → Replan → Research → Progress → Ask.

## Render deployment

Build:

```text
pip install -r requirements.txt
```

Start:

```text
python app.py
```

Required secret:

```text
GROQ_API_KEY
```

Never commit the API key to GitHub.

## Environment variables

```text
FAST_MODEL_NAME=groq/qwen/qwen3.8-27b
DEEP_MODEL_NAME=groq/openai/gpt-oss-120b
```

## Important

Learner state is session-based for this hackathon prototype. It is not a permanent database or account system.
