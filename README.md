---
title: EduPath
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
python_version: "3.10"
---

# 🎓 EduPath — Adaptive Learning Companion

EduPath is an agentic AI learning and skill-gap application.

It creates a personalized learning journey from a learner's current skills and target role, generates practice, assesses submitted work, updates mastery, recalculates gaps, replans the journey, researches learning resources, and produces progress reports.

## Core flow

**Profile → Skill Gaps → Plan → Practice → Assessment → Adapt → Research → Progress**

## Stack

- Google ADK
- Groq hosted LLMs
- LiteLLM
- Gradio
- Pydantic
- Groq browser search for live resource research

## Hugging Face setup

1. Create a **Public Gradio Space**.
2. Upload `app.py`, `requirements.txt`, and this `README.md`.
3. Open **Settings → Secrets**.
4. Add the secret:

`GROQ_API_KEY`

5. Let the Space rebuild.
6. Open the Space app.

Do **not** put the Groq API key in `app.py`, `README.md`, or GitHub.

## Important

The app uses session-based learner state for the prototype. It does not implement permanent user accounts or a production database.

The research stage intentionally separates web search from structured resource curation.

## Demo suggestion

Use one consistent learner persona:

- 4 years automotive / EV engineering
- strong PMSM, FOC, MATLAB, Simulink
- developing Python / ML / Agentic AI skills
- target role: AI-enabled EV Controls Engineer

Then demonstrate:

1. Analyze Profile
2. Generate Plan
3. Generate Practice
4. Submit Work
5. Assess & Adapt
6. Generate Updated Plan
7. Research Priority Gap
8. Generate Progress Report
9. Ask EduPath
