# EduPath — Final Two-LLM + Canonical Skill Strategy

## Two LLMs

**Qwen 3.8 27B**
- compact learner profile extraction
- learning plan
- practice task
- learner Q&A

**GPT-OSS 120B**
- deep assessment, split into two compact stages

## Critical profile fix

The UI already owns target role, experience years and weekly hours.
The profile LLM therefore returns only goals and skills.

A deterministic Python canonicalization layer then converts model labels into
the internal skill vocabulary.

Examples:

- `MATLAB & Simulink` → `MATLAB`, `Simulink`
- `Python & Basic ML` → `Python`, `Machine Learning`
- `LLMs & Agentic AI` → `LLMs`, `Agentic AI`
- `Field-Oriented Control (FOC)` → `FOC`

This prevents valid learner evidence from being lost because the LLM used a
different spelling or combined two related skills.

The system does NOT automatically map generic `motor control` to `Control Systems`,
because that would create unsupported evidence.

## Deterministic ownership

Python owns:
- UI-authoritative learner fields
- skill canonicalization
- gap calculation
- mastery updates
- progress calculation
