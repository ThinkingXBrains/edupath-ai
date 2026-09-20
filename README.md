---
title: EduPath
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
python_version: "3.10"
---

# 🎓 EduPath — Adaptive Learning Companion

EduPath is an agentic AI learning and skill-gap application.

## Core flow

**Profile → Skill Gaps → Plan → Practice → Assessment → Adapt → Research → Progress → Ask**

## Stack

- Google ADK
- Groq
- LiteLLM
- Gradio
- Pydantic
- pypdf

## Run locally

```bash
pip install -r requirements.txt
set GROQ_API_KEY=YOUR_KEY
python app.py
```

On Linux/macOS:

```bash
export GROQ_API_KEY=YOUR_KEY
python app.py
```

## Render deployment

Create a Render **Web Service** connected to this GitHub repository.

Build Command:

```text
pip install -r requirements.txt
```

Start Command:

```text
python app.py
```

Plan:

```text
Free
```

Add the environment variable:

```text
GROQ_API_KEY
```

Do not commit the API key to GitHub.

## Hugging Face

Ordinary Gradio Spaces may require paid access depending on the account. This repository is intentionally deployment-neutral and works as a Python/Gradio web service on Render.

## Security

Keep `GROQ_API_KEY` only in the hosting provider's secret/environment-variable store.


## Render port binding

The app explicitly binds Gradio to `0.0.0.0` and the Render-provided `PORT`.
SSR is disabled for the Render deployment to keep the server path simple.
