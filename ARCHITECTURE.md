# EduPath Architecture

```text
                       ┌───────────────────┐
                       │      Learner      │
                       └─────────┬─────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │    Profile Agent    │
                      │  Qwen 3.8 27B       │
                      └─────────┬───────────┘
                                │
                                ▼
                       Deterministic Gap Engine
                                │
                                ▼
                      ┌─────────────────────┐
                      │    Planner Agent    │
                      │  Qwen 3.8 27B       │
                      └─────────┬───────────┘
                                │
                                ▼
                      ┌─────────────────────┐
                      │   Practice Agent    │
                      │  Qwen 3.8 27B       │
                      └─────────┬───────────┘
                                │
                                ▼
                         Learner Submission
                                │
                                ▼
                      ┌─────────────────────┐
                      │ Deep Assessment     │
                      │ GPT-OSS 120B        │
                      └─────────┬───────────┘
                                │
                       rate-limit fallback
                                │
                         ┌──────▼──────┐
                         │ Qwen 27B    │
                         └──────┬──────┘
                                │
                                ▼
                        Deterministic Mastery
                           + Gap Update
                                │
                                ▼
                            Re-planner
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
             Web Search     Progress        Q&A
               (DDGS)      Qwen 27B      Qwen 27B
                  │
                  ▼
            Resource Curator
              Qwen 27B
```

### Why two models?

Use the compact Qwen model for high-frequency, structured and conversational tasks.

Reserve GPT-OSS 120B for the highest-value deep assessment stage, with a Qwen fallback.

### What is deterministic?

- skill-name normalization
- target-role skill levels
- gap calculation
- mastery update
- state management

The LLMs do not directly decide the numerical mastery update rule.
