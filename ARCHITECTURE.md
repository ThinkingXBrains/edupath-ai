# EduPath architecture

```text
Learner
  ↓
Profile Insights — Qwen 3.8 27B
  ↓
Python canonicalization
  ↓
Deterministic skill gaps
  ↓
Plan — Qwen 3.8 27B
  ↓
Practice — Qwen 3.8 27B
  ↓
Learner submission
  ↓
Deep notes — GPT-OSS 120B
  ↓
Compact notes
  ↓
Final assessment — GPT-OSS 120B
  ↓
Deterministic mastery update
  ↓
Gap recalculation
  ↓
Replan — Qwen 3.8 27B
  ↓
Research — DDGS
  ↓
Progress — Python
  ↓
Ask EduPath — Qwen 3.8 27B
```

The ADK/LiteLLM/model layer is imported lazily so Render can bind its web port
before any model objects are constructed.
