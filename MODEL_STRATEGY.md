# EduPath Model Strategy

**Structured agents:** GPT-OSS 120B through Google ADK/LiteLLM for strict Pydantic outputs.

**Research + Ask EduPath:** Groq Compound Mini through the Groq SDK for web-heavy and conversational tasks. Compound provides server-side web search and has a higher listed TPM allowance than the GPT-OSS model rows in Groq's rate-limit table.

**Rate-limit handling:** ADK calls retry once when Groq provides a retry interval.
