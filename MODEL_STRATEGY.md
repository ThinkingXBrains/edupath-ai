# Final model strategy

The application deliberately separates inference from deterministic state.

Qwen 3.8 27B is the high-frequency workhorse. GPT-OSS 120B is reserved for the
highest-value deep assessment and is used in two compact stages. The second
stage receives only the compact first-stage notes, not the full learner
submission.

The profile schema is intentionally small. Role, experience and weekly hours
come from the UI; the model only returns goals and evidence-backed skills.

The application uses a client-side rolling output reservation to avoid bursts
against the current account's output-token allowance. This does not circumvent
provider limits.
