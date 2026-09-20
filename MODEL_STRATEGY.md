# EduPath — Token-Safe Two-LLM Strategy

## The profile error

The previous profile schema required the model to generate target_role,
experience_years and weekly_hours even though the UI already had those values.
The completion was truncated before the required fields.

## Fix

Qwen now returns only:
- up to 3 goals
- up to 6 evidence-backed skills

Python supplies:
- target role
- experience years
- weekly hours

and constructs the full LearnerProfile.

## GPT-OSS 120B deep assessment

Two compact stages:

1. task + submission → compact AssessmentNotes
2. AssessmentNotes only → final AssessmentResult

Stage 2 never receives the original learner submission.

## Token discipline

- explicit output-token caps
- compact prompts
- compact schemas
- local rolling output-token reservation
- research through DDGS
- progress and mastery updates in Python

This is token control, not a mechanism for bypassing Groq's provider limits.
