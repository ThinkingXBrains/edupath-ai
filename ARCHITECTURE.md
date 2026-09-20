# EduPath Architecture

```text
Learner
  ↓
Profile Agent — Qwen 3.8 27B
  ↓
Deterministic Skill Gap Engine
  ↓
Planner — Qwen 3.8 27B
  ↓
Practice — Qwen 3.8 27B
  ↓
Submission
  ↓
Deep Assessment Stage 1 — GPT-OSS 120B
  ↓
Compact notes
  ↓
Deep Assessment Stage 2 — GPT-OSS 120B
  ↓
Mastery update — Python
  ↓
Gap recalculation — Python
  ↓
Replan — Qwen 3.8 27B
  ↓
Research — DDGS
  ↓
Progress — Python
  ↓
Ask EduPath — Qwen 3.8 27B
```

The LLMs interpret evidence and generate learning content. Python owns numerical state transitions and mastery updates.
