---
title: EduPath
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
---

# EduPath — Adaptive Learning Companion

Two-model learning agent built with Google ADK + Groq.

## Model routing

**Qwen 3.8 27B**
- learner profile insights
- learning plan
- practice task
- learner Q&A

**GPT-OSS 120B**
- deep practice assessment
- Stage 1: compact evidence analysis
- Stage 2: compact final structured result

**Python**
- canonical skill mapping
- gap calculation
- mastery update
- progress report

**DDGS**
- web research without an additional LLM call

## Important design choice

The profile model does not regenerate UI-owned fields such as target role,
experience years or weekly hours. This prevents unnecessary output tokens and
schema truncation.

## Render

Build command:

`pip install -r requirements.txt`

Start command:

`python app.py`

Required secret:

`GROQ_API_KEY`
