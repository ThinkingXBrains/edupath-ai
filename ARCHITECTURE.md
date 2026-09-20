# EduPath Architecture

```text
                         ┌───────────────────────┐
                         │       Learner         │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     Profile Agent     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Deterministic Gap   │
                         │        Engine         │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     Planner Agent     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Practice Agent     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Assessment Agent    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  Skill State Update   │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     Recalculate       │
                         │       Gaps            │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │      Re-planner       │
                         └───────────┬───────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
       ┌────────────────┐   ┌─────────────────┐  ┌────────────────┐
       │ Research Agent │   │ Progress Agent  │  │   Ask Agent    │
       └───────┬────────┘   └─────────────────┘  └────────────────┘
               │
               ▼
       ┌────────────────┐
       │ Resource       │
       │ Curator        │
       └────────────────┘
```

### Design principle

LLMs handle interpretation, planning, assessment and research.

Deterministic Python handles skill normalization, gap calculation, mastery updates and session state.
