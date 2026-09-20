import os
import json
import uuid
import asyncio
import time
import re
from pathlib import Path
from typing import Any

import gradio as gr
from pydantic import BaseModel, Field
from groq import Groq
from ddgs import DDGS

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "edupath"

# Hybrid two-LLM strategy
FAST_MODEL_NAME = os.getenv(
    "FAST_MODEL_NAME",
    "groq/qwen/qwen3.8-27b",
)
DEEP_MODEL_NAME = os.getenv(
    "DEEP_MODEL_NAME",
    "groq/openai/gpt-oss-120b",
)

# Conservative local rolling output-token reservations.
# These values are below the 1K output-token/minute ceiling
# shown in the user's current Groq error.
FAST_OUTPUT_BUDGET_PER_MINUTE = 850
DEEP_OUTPUT_BUDGET_PER_MINUTE = 700

_fast_usage = []
_deep_usage = []

_fast_budget_lock = asyncio.Lock()
_deep_budget_lock = asyncio.Lock()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to your Render Environment Variables."
    )

groq_client = Groq(
    api_key=GROQ_API_KEY,
    default_headers={"Groq-Model-Version": "latest"},
)


def fast_model():
    # Qwen 3.8 27B supports strict structured outputs and tunable reasoning.
    return LiteLlm(
        model=FAST_MODEL_NAME,
        api_key=GROQ_API_KEY,
        reasoning_effort="none",
        include_reasoning=False,
    )


def deep_model():
    # GPT-OSS 120B is reserved for the hardest assessment stage.
    return LiteLlm(
        model=DEEP_MODEL_NAME,
        api_key=GROQ_API_KEY,
        reasoning_effort="low",
        include_reasoning=False,
    )


# ============================================================
# ROLE KNOWLEDGE
# ============================================================

ROLE_SKILLS = {
    "AI-enabled EV Controls Engineer": {
        "Control Systems": 0.90,
        "FOC": 0.90,
        "PMSM": 0.85,
        "MATLAB": 0.85,
        "Simulink": 0.85,
        "Power Electronics": 0.85,
        "Python": 0.75,
        "Data Analysis": 0.70,
        "Machine Learning": 0.65,
        "LLMs": 0.65,
        "Agentic AI": 0.75,
        "MCP": 0.60,
    },
    "AI Engineer": {
        "Python": 0.85,
        "Data Analysis": 0.75,
        "Machine Learning": 0.80,
        "Deep Learning": 0.70,
        "LLMs": 0.80,
        "Agentic AI": 0.75,
        "APIs": 0.70,
    },
}


SKILL_ALIASES = {
    "power electronics engineering": "Power Electronics",
    "power electronics": "Power Electronics",

    "permanent magnet synchronous motors": "PMSM",
    "permanent magnet synchronous motor": "PMSM",
    "permanent magnet synchronous motors (pmsm)": "PMSM",
    "permanent magnet synchronous motor (pmsm)": "PMSM",
    "pmsm": "PMSM",

    "field oriented control": "FOC",
    "field-oriented control": "FOC",
    "field oriented control (foc)": "FOC",
    "field-oriented control (foc)": "FOC",
    "foc": "FOC",

    "space vector pwm": "SVPWM",
    "space vector pwm (svpwm)": "SVPWM",
    "svpwm": "SVPWM",

    "machine learning": "Machine Learning",
    "machine learning (basic)": "Machine Learning",
    "basic machine learning": "Machine Learning",

    "large language model": "LLMs",
    "large language models": "LLMs",
    "llm": "LLMs",
    "llms": "LLMs",

    "agentic ai": "Agentic AI",

    "data analysis": "Data Analysis",
    "data analytics": "Data Analysis",

    "python": "Python",
    "python programming": "Python",

    "deep learning": "Deep Learning",
    "matlab": "MATLAB",
    "simulink": "Simulink",
    "mcp": "MCP",
    "apis": "APIs",
    "control systems": "Control Systems",
    "control system": "Control Systems",
    "torque control": "Torque control",
    "regenerative braking": "Regenerative braking",
    "bldc": "BLDC",
    "bldc motor": "BLDC",
    "bldc motors": "BLDC",
}


# Explicitly handle the compound labels that small LLMs
# commonly produce when summarising related skills.
COMPOUND_SKILL_EXPANSIONS = {
    "matlab & simulink": ["MATLAB", "Simulink"],
    "matlab and simulink": ["MATLAB", "Simulink"],
    "python & basic ml": ["Python", "Machine Learning"],
    "python and basic ml": ["Python", "Machine Learning"],
    "python & machine learning": ["Python", "Machine Learning"],
    "python and machine learning": ["Python", "Machine Learning"],
    "llms & agentic ai": ["LLMs", "Agentic AI"],
    "llms and agentic ai": ["LLMs", "Agentic AI"],
    "llm & agentic ai": ["LLMs", "Agentic AI"],
    "llm and agentic ai": ["LLMs", "Agentic AI"],
    # Do NOT map "motor control" to Control Systems automatically:
    # that would be an unsupported inference.
    "power electronics & motor control": ["Power Electronics"],
    "power electronics and motor control": ["Power Electronics"],
}


PHRASE_TO_SKILL = [
    ("field-oriented control", "FOC"),
    ("field oriented control", "FOC"),
    ("space vector pwm", "SVPWM"),
    ("permanent magnet synchronous motor", "PMSM"),
    ("power electronics", "Power Electronics"),
    ("regenerative braking", "Regenerative braking"),
    ("machine learning", "Machine Learning"),
    ("data analysis", "Data Analysis"),
    ("agentic ai", "Agentic AI"),
    ("deep learning", "Deep Learning"),
    ("simulink", "Simulink"),
    ("matlab", "MATLAB"),
    ("python", "Python"),
    ("llms", "LLMs"),
    ("llm", "LLMs"),
    ("mcp", "MCP"),
    ("control systems", "Control Systems"),
    ("control system", "Control Systems"),
    ("pmsm", "PMSM"),
    ("svpwm", "SVPWM"),
    ("foc", "FOC"),
    ("bldc", "BLDC"),
    ("apis", "APIs"),
]


def _normalise_text(value: str) -> str:
    return " ".join(
        value.strip().lower().split()
    )


def normalize_skill_name(name: str) -> str:
    key = _normalise_text(name)

    return SKILL_ALIASES.get(
        key,
        name.strip(),
    )


def expand_skill_name(name: str) -> list[str]:
    """
    Convert one LLM label into one or more internal canonical skills.

    This is intentionally deterministic. It prevents labels such as
    'MATLAB & Simulink' from becoming a single unmatched skill key.
    """

    key = _normalise_text(name)

    if key in COMPOUND_SKILL_EXPANSIONS:
        return list(
            COMPOUND_SKILL_EXPANSIONS[key]
        )

    exact = SKILL_ALIASES.get(key)

    if exact:
        return [exact]

    found = []

    for phrase, canonical in PHRASE_TO_SKILL:

        if phrase in {
            "foc",
            "svpwm",
            "pmsm",
            "llm",
            "llms",
            "mcp",
            "bldc",
            "apis",
        }:

            if re.search(
                rf"\\b{re.escape(phrase)}\\b",
                key,
            ):

                if canonical not in found:
                    found.append(canonical)

        elif phrase in key:

            if canonical not in found:
                found.append(canonical)

    return found or [name.strip()]


def canonicalize_profile_skills(
    skills: list[Skill],
) -> list[Skill]:
    """
    Merge LLM labels into canonical skill records.

    Example:
      'MATLAB & Simulink' →
          MATLAB + Simulink

      'Python & Basic ML' →
          Python + Machine Learning
    """

    merged: dict[str, dict[str, Any]] = {}

    for skill in skills:

        names = expand_skill_name(
            skill.name
        )

        for canonical in names:

            if canonical not in merged:

                merged[canonical] = {
                    "name": canonical,
                    "proficiency": float(
                        skill.proficiency
                    ),
                    "confidence": float(
                        skill.confidence
                    ),
                    "evidence": list(
                        skill.evidence
                    ),
                }

            else:

                record = merged[
                    canonical
                ]

                record[
                    "proficiency"
                ] = max(
                    record[
                        "proficiency"
                    ],
                    float(
                        skill.proficiency
                    ),
                )

                record[
                    "confidence"
                ] = max(
                    record[
                        "confidence"
                    ],
                    float(
                        skill.confidence
                    ),
                )

                for evidence in skill.evidence:

                    if evidence not in record[
                        "evidence"
                    ]:

                        record[
                            "evidence"
                        ].append(
                            evidence
                        )

    return [
        Skill.model_validate(
            record
        )
        for record in merged.values()
    ]


def calculate_skill_gaps(
    profile,
    target_role: str,
):
    target_skills = ROLE_SKILLS[
        target_role
    ]

    current = {
        normalize_skill_name(
            skill.name
        ): skill
        for skill in profile.skills
    }

    gaps = []

    for skill_name, target in target_skills.items():

        item = current.get(
            skill_name
        )

        if item:

            mastery = float(
                item.proficiency
            )

            confidence = float(
                item.confidence
            )

        else:

            mastery = 0.0
            confidence = 0.0

        gaps.append(
            {
                "skill": skill_name,
                "current": round(
                    mastery,
                    2,
                ),
                "target": round(
                    target,
                    2,
                ),
                "gap": round(
                    max(
                        target - mastery,
                        0.0,
                    ),
                    2,
                ),
                "confidence": round(
                    confidence,
                    2,
                ),
            }
        )

    gaps.sort(
        key=lambda x: x["gap"],
        reverse=True,
    )

    return gaps


# ============================================================
# SCHEMAS
# ============================================================

class Skill(BaseModel):
    name: str
    proficiency: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]


class ProfileInsights(BaseModel):
    goals: list[str] = Field(max_length=3)
    skills: list[Skill] = Field(max_length=6)


class LearnerProfile(BaseModel):
    target_role: str
    experience_years: float
    goals: list[str]
    weekly_hours: float
    skills: list[Skill]


class LearningWeek(BaseModel):
    week: int
    objective: str
    topics: list[str]
    practice_tasks: list[str]
    deliverable: str
    assessment: str
    estimated_hours: float


class LearningPlan(BaseModel):
    target_role: str
    total_weeks: int
    weekly_hours: float
    weeks: list[LearningWeek]


class PracticeTask(BaseModel):
    skill: str
    title: str
    difficulty: str
    objective: str
    instructions: list[str]
    deliverable: str
    estimated_hours: float
    success_criteria: list[str]


class AssessmentNotes(BaseModel):
    skill: str
    score_estimate: float = Field(ge=0.0, le=1.0)
    strengths: list[str] = Field(max_length=2)
    weaknesses: list[str] = Field(max_length=2)
    evidence: list[str] = Field(max_length=2)
    key_reason: str


class AssessmentResult(BaseModel):
    skill: str
    score: float = Field(ge=0.0, le=1.0)
    strengths: list[str] = Field(max_length=2)
    weaknesses: list[str] = Field(max_length=2)
    evidence: list[str] = Field(max_length=2)
    feedback: str
    recommended_action: str


class LearningResource(BaseModel):
    title: str
    provider: str
    url: str
    relevance: str
    difficulty: str
    estimated_hours: float
    cost: str


class ResourceCollection(BaseModel):
    skill: str
    resources: list[LearningResource]


class SkillProgress(BaseModel):
    skill: str
    mastery: float
    status: str
    evidence: list[str]


class ProgressReport(BaseModel):
    learner_role: str
    plan_version: int
    skills_acquired: list[SkillProgress]
    skills_in_progress: list[SkillProgress]
    remaining_gaps: list[str]
    recommended_next_steps: list[str]
    summary: str


# ============================================================
# AGENTS — HYBRID ROUTING
# ============================================================

# FAST / EFFICIENT MODEL:
# Qwen 3.8 27B is used for most structured and conversational work.
profile_agent = LlmAgent(
    name="profile_agent",
    model=fast_model(),
    description="Extracts compact learner insights.",
    instruction="""
Extract ONLY learner goals and evidence-backed skills.

The UI already supplies:
- target role
- experience years
- weekly learning hours

DO NOT output those fields.

Use these canonical skill names whenever applicable:
Control Systems, FOC, PMSM, SVPWM, MATLAB, Simulink,
Power Electronics, Python, Data Analysis, Machine Learning,
LLMs, Agentic AI, MCP, Deep Learning, APIs,
Regenerative Braking, BLDC, Torque control.

Examples:
- "field oriented control" -> "FOC"
- "permanent magnet synchronous motor" -> "PMSM"
- "MATLAB & Simulink" -> prefer two separate skills
- "Python & Basic ML" -> prefer two separate skills
- "LLMs & Agentic AI" -> prefer two separate skills

Rules:
- maximum 6 source skill labels
- maximum 3 goals
- one short evidence sentence per skill
- do not invent skills
- do not combine unrelated skills
- no explanation
- return only structured output
""",
    output_schema=ProfileInsights,
    output_key="profile_insights",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=320,
    ),
)



planner_agent = LlmAgent(
    name="planner_agent",
    model=fast_model(),
    description="Builds a personalized four-week learning plan.",
    instruction="""
You are EduPath's Learning Planner.

Create a realistic 4-week learning plan based on the learner's:
- target role
- current skills
- proficiency
- confidence
- remaining gaps
- weekly learning time

Prioritize meaningful gaps.
Leverage existing domain strengths.
Avoid spending large amounts of time reteaching demonstrated skills.
Make each week practical and measurable.

Return only the structured output.
""",
    output_schema=LearningPlan,
    output_key="learning_plan",
)


practice_agent = LlmAgent(
    name="practice_agent",
    model=fast_model(),
    description="Creates a practical hands-on task for the learner's current gap.",
    instruction="""
You are EduPath's Practice Task Generator.

Create one practical, project-oriented task for the learner's priority gap.

Match difficulty to the learner's current level.
Leverage their existing domain knowledge when useful.
Keep it achievable within the available weekly time.
Provide a concrete deliverable and measurable success criteria.

Return only the structured output.
""",
    output_schema=PracticeTask,
    output_key="practice_task",
)


# DEEP MODEL:
# GPT-OSS 120B is reserved for the highest-value reasoning step.
assessment_agent_deep = LlmAgent(
    name="assessment_agent_deep",
    model=deep_model(),
    description="GPT-OSS 120B stage 1: compact assessment notes.",
    instruction="""
Analyze the practice task and learner submission.

Return COMPACT NOTES ONLY.

Limits:
- maximum 2 strengths
- maximum 2 weaknesses
- maximum 2 evidence items
- key_reason = one short sentence
- score_estimate = 0.0 to 1.0
- no long explanation
- use only submitted evidence

Return only the structured output.
""",
    output_schema=AssessmentNotes,
    output_key="assessment_notes",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=180,
    ),
)


assessment_agent_fast = LlmAgent(
    name="assessment_agent_fast",
    model=fast_model(),
    description="Fallback compact assessment.",
    instruction="""
Assess the submission using only supplied evidence.

Limits:
- maximum 2 strengths
- maximum 2 weaknesses
- maximum 2 evidence items
- feedback = one short paragraph
- recommended_action = one short sentence
- no invented evidence

Return only the structured output.
""",
    output_schema=AssessmentResult,
    output_key="assessment_result",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=260,
    ),
)


assessment_final_agent = LlmAgent(
    name="assessment_final_agent",
    model=deep_model(),
    description="GPT-OSS 120B stage 2: compact notes to final assessment.",
    instruction="""
Convert ONLY the supplied compact assessment notes into the final assessment.

Do not analyze the original submission.
Do not add new evidence.

Limits:
- maximum 2 strengths
- maximum 2 weaknesses
- maximum 2 evidence items
- feedback = one short paragraph
- recommended_action = one short sentence

Return only the structured output.
""",
    output_schema=AssessmentResult,
    output_key="assessment_result",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=260,
    ),
)


resource_curator_agent = LlmAgent(
    name="resource_curator_agent",
    model=fast_model(),
    description="Converts web-search evidence into curated learning resources.",
    instruction="""
You are EduPath's Resource Curator.

You receive web-search results and verified URLs.

Select 3-4 resources that best match the learner's skill gap and level.

Rules:
1. Use only URLs present in the supplied search results.
2. Do not invent or alter URLs.
3. Remove duplicates.
4. Prefer official documentation, universities and established learning platforms.
5. Return only the structured ResourceCollection.
""",
    output_schema=ResourceCollection,
    output_key="resource_collection",
)


progress_agent = LlmAgent(
    name="progress_agent",
    model=fast_model(),
    description="Generates a concise progress report from learner state.",
    instruction="""
You are EduPath's Progress Analyst.

Use the supplied learner state, assessments and gaps.

Classify:
- Acquired: strong demonstrated mastery
- In Progress: evidence exists but target is not reached

Also report remaining gaps and specific next steps.

Do not invent achievements.
Return only the structured output.
""",
    output_schema=ProgressReport,
    output_key="progress_report",
)


qa_agent = LlmAgent(
    name="qa_agent",
    model=fast_model(),
    description="Answers learner questions using current EduPath state.",
    instruction="""
You are EduPath's Learning Companion.

Answer the learner using the supplied learner state.
Be practical and concise.
Do not invent scores, achievements, skills or resources.
""",
)


# ============================================================
# ADK RUNNER
# ============================================================

def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "rate limit" in text
        or "rate_limit" in text
        or "tokens per minute" in text
        or "429" in text
    )


async def reserve_output_budget(
    bucket,
    lock,
    limit,
    estimated_tokens,
):
    while True:

        async with lock:

            now = time.monotonic()
            cutoff = now - 60.0

            bucket[:] = [
                (timestamp, amount)
                for timestamp, amount in bucket
                if timestamp >= cutoff
            ]

            used = sum(
                amount
                for _, amount in bucket
            )

            if used + estimated_tokens <= limit:

                bucket.append(
                    (
                        now,
                        estimated_tokens,
                    )
                )

                return

            wait_for = (
                bucket[0][0]
                + 60.0
                - now
                + 0.2
            )

        await asyncio.sleep(
            max(
                wait_for,
                0.2,
            )
        )


async def run_adk_agent(
    agent,
    message_text: str,
    output_key: str | None = None,
    user_prefix: str = "edupath",
    estimated_output_tokens=200,
    budget="fast",
):
    """
    Run an ADK agent in a short-lived in-memory session.
    """
    # Local rolling output reservation. This does not bypass Groq limits;
    # it simply spaces our small requests apart.
    if budget == "deep":
        await reserve_output_budget(
            _deep_usage,
            _deep_budget_lock,
            DEEP_OUTPUT_BUDGET_PER_MINUTE,
            estimated_output_tokens,
        )
    else:
        await reserve_output_budget(
            _fast_usage,
            _fast_budget_lock,
            FAST_OUTPUT_BUDGET_PER_MINUTE,
            estimated_output_tokens,
        )

    user_id = f"{user_prefix}_{uuid.uuid4().hex[:8]}"

    runner = InMemoryRunner(
        agent=agent,
        app_name=APP_NAME,
    )

    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
    )

    message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=message_text
            )
        ],
    )

    events = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=message,
    ):
        events.append(event)

    if output_key:
        state = await runner.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session.id,
        )

        result = state.state.get(output_key)

        if isinstance(result, str):
            result = json.loads(result)

        return result

    text_parts = []

    for event in events:
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    text_parts.append(part.text)

    return "\n".join(text_parts)


async def run_with_fallback(
    primary_agent,
    fallback_agent,
    message_text,
    output_key,
    user_prefix,
):
    try:
        return await run_adk_agent(
            primary_agent,
            message_text,
            output_key=output_key,
            user_prefix=user_prefix,
        )
    except Exception as primary_error:
        # Fallback is intentionally used for quota/rate-limit errors.
        if not is_rate_limit_error(primary_error):
            raise

        return await run_adk_agent(
            fallback_agent,
            message_text,
            output_key=output_key,
            user_prefix=f"{user_prefix}_fallback",
        )


# ============================================================
# FORMATTERS
# ============================================================

def format_profile(
    profile: LearnerProfile
) -> str:

    goals = "\n".join(
        f"- {goal}"
        for goal in profile.goals
    ) or "- —"

    detected = "\n".join(
        f"- ✅ {skill.name} — "
        f"{skill.proficiency:.0%} "
        f"(confidence {skill.confidence:.0%})"
        for skill in profile.skills
    ) or "- No evidence-backed skills detected."

    return f"""
## 👤 Learner Profile

**Target role**  
{profile.target_role}

**Experience**  
{profile.experience_years:.1f} years

**Learning time**  
{profile.weekly_hours:.1f} hrs/week

### Goals
{goals}

### Detected Skills
{detected}
"""


def format_gaps(gaps, top_n=None) -> str:
    items = gaps[:top_n] if top_n else gaps

    rows = []

    for item in items:
        current = item["current"]
        target = item["target"]
        filled = int(current * 20)
        bar = "█" * filled + "░" * (20 - filled)

        rows.append(
            f"{item['skill']:25s} "
            f"{bar} "
            f"{int(current * 100):>3}% → {int(target * 100):>3}%"
        )

    if not rows:
        return "No skill gaps available."

    return "```text\n" + "\n".join(rows) + "\n```"


def format_plan(plan: LearningPlan) -> str:
    text = f"""
## 📚 Personalized Learning Plan

**{plan.target_role}** · **{plan.total_weeks} weeks** · **{plan.weekly_hours:.1f} hrs/week**
"""

    for week in plan.weeks:
        topics = "\n".join(
            f"- {x}"
            for x in week.topics
        )

        practice = "\n".join(
            f"- {x}"
            for x in week.practice_tasks
        )

        text += f"""
### Week {week.week}

**Objective**  
{week.objective}

**Topics**  
{topics}

**Practice**  
{practice}

**Deliverable**  
{week.deliverable}

**Assessment**  
{week.assessment}

**Estimated time:** {week.estimated_hours:.1f} hours

"""

    return text


def format_task(task: PracticeTask) -> str:
    instructions = "\n".join(
        f"{i}. {x}"
        for i, x in enumerate(task.instructions, 1)
    )

    criteria = "\n".join(
        f"- {x}"
        for x in task.success_criteria
    )

    return f"""
## 🛠️ Current Practice Task

### {task.title}

**Skill:** {task.skill} · **{task.difficulty}** · **{task.estimated_hours:.1f} hrs**

### Objective
{task.objective}

### Instructions
{instructions}

### Deliverable
{task.deliverable}

### Success Criteria
{criteria}
"""


def format_assessment(result: AssessmentResult) -> str:
    strengths = "\n".join(
        f"- {x}"
        for x in result.strengths
    )

    weaknesses = "\n".join(
        f"- {x}"
        for x in result.weaknesses
    )

    return f"""
## 📝 Assessment

### {result.score:.0%} · {result.skill}

**Strengths**
{strengths}

**Areas to improve**
{weaknesses}

### Feedback
{result.feedback}

### Next action
{result.recommended_action}
"""


def format_resources(collection: ResourceCollection) -> str:
    text = f"""
## 🔎 Recommended Resources

**Current focus:** {collection.skill}
"""

    for i, resource in enumerate(
        collection.resources,
        1,
    ):
        text += f"""
### {i}. {resource.title}

**{resource.provider}** · {resource.difficulty} · {resource.estimated_hours:.1f} hrs · {resource.cost}

{resource.relevance}

🔗 {resource.url}

"""

    return text


def format_progress(report: ProgressReport) -> str:
    acquired = "\n".join(
        f"- ✅ {x.skill} — {x.mastery:.0%}"
        for x in report.skills_acquired
    ) or "- None yet"

    in_progress = "\n".join(
        f"- 🔄 {x.skill} — {x.mastery:.0%}"
        for x in report.skills_in_progress
    ) or "- None yet"

    remaining = "\n".join(
        f"- {x}"
        for x in report.remaining_gaps
    ) or "- None"

    next_steps = "\n".join(
        f"- {x}"
        for x in report.recommended_next_steps
    ) or "- None"

    return f"""
## 📈 Progress Report

**Plan version:** {report.plan_version}

### Summary
{report.summary}

### Acquired
{acquired}

### In progress
{in_progress}

### Remaining gaps
{remaining}

### Recommended next steps
{next_steps}
"""


# ============================================================
# RESEARCH TOOL — NO SECOND LLM NEEDED
# ============================================================

def web_search(query: str, max_results: int = 6) -> list[dict[str, Any]]:
    """
    Search the public web through DDGS.
    Returns title, url and snippet for downstream curation.
    """
    results = DDGS().text(
        query,
        max_results=max_results,
        backend="auto",
    )

    clean = []

    for item in results:
        clean.append(
            {
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "snippet": item.get("body", ""),
            }
        )

    return clean


# ============================================================
# UI CALLBACKS
# ============================================================

async def ui_analyze_profile(
    target_role,
    experience_years,
    weekly_hours,
    learner_background,
    resume_file,
    ui_state,
):
    try:

        resume_text = ""

        if resume_file:

            path = resume_file

            if path.lower().endswith(".pdf"):

                from pypdf import PdfReader

                reader = PdfReader(path)

                resume_text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )

            elif path.lower().endswith(".txt"):

                resume_text = Path(
                    path
                ).read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

        # The LLM only infers goals and skills.
        # Role, experience and hours come directly from the UI.
        prompt = f"""
LEARNER:
{learner_background[:5000]}

RESUME:
{resume_text[:5000]}
"""

        raw = await run_adk_agent(
            profile_agent,
            prompt,
            output_key="profile_insights",
            user_prefix="profile",
        )

        insights = ProfileInsights.model_validate(
            raw
        )

        # Repair / canonicalize the model's skill labels
        # before calculating any gaps.
        canonical_skills = (
            canonicalize_profile_skills(
                insights.skills
            )
        )

        profile = LearnerProfile(
            target_role=target_role,
            experience_years=float(
                experience_years
            ),
            goals=insights.goals,
            weekly_hours=float(
                weekly_hours
            ),
            skills=canonical_skills,
        )

        gaps = calculate_skill_gaps(
            profile,
            target_role,
        )

        skills = {}

        for skill in profile.skills:

            name = normalize_skill_name(
                skill.name
            )

            skills[name] = {
                "mastery": skill.proficiency,
                "confidence": skill.confidence,
                "evidence": list(
                    skill.evidence
                ),
                "assessment_scores": [],
            }

        state = {
            "target_role": target_role,
            "weekly_hours": float(
                weekly_hours
            ),
            "profile": profile.model_dump(),
            "skills": skills,
            "gaps": gaps,
            "plan_version": 1,
            "learning_plan": None,
            "current_task": None,
            "completed_tasks": [],
            "latest_assessment": None,
            "recommended_resources": [],
        }

        return (
            format_profile(profile),
            "## 🎯 Skill Gap Map\n\n"
            + format_gaps(gaps),
            state,
        )

    except Exception as exc:

        return (
            f"❌ **Profile analysis failed**\n\n"
            f"`{type(exc).__name__}: {exc}`",
            "",
            ui_state,
        )




async def ui_generate_plan(ui_state):
    try:
        if not ui_state:
            return "⚠️ Analyze your profile first.", ui_state

        profile = ui_state["profile"]
        gaps = ui_state["gaps"]

        prompt = f"""
TARGET ROLE:
{profile["target_role"]}

EXPERIENCE:
{profile["experience_years"]} years

WEEKLY HOURS:
{profile["weekly_hours"]}

CURRENT SKILLS:
{json.dumps(profile["skills"][:6], separators=(",", ":"))}

CURRENT GAPS:
{json.dumps(gaps[:6], separators=(",", ":"))}

Create a realistic four-week plan.
"""

        raw_plan = await run_adk_agent(
            planner_agent,
            prompt,
            output_key="learning_plan",
            user_prefix="planner",
            estimated_output_tokens=420,
            budget="fast",
        )

        plan = LearningPlan.model_validate(
            raw_plan
        )

        ui_state["learning_plan"] = plan.model_dump()

        return (
            format_plan(plan),
            ui_state,
        )

    except Exception as exc:
        return (
            f"❌ **Plan generation failed**\n\n`{type(exc).__name__}: {exc}`",
            ui_state,
        )


async def ui_generate_practice(ui_state):
    try:
        if not ui_state:
            return "⚠️ Analyze your profile first.", ui_state

        gaps = ui_state.get("gaps", [])

        if not gaps:
            return "⚠️ No current gaps are available.", ui_state

        priority = gaps[0]

        prompt = f"""
TARGET ROLE:
{ui_state["target_role"]}

WEEKLY HOURS:
{ui_state["weekly_hours"]}

PRIORITY GAP:
{json.dumps(priority, separators=(",", ":"))}

CURRENT SKILLS:
{json.dumps(ui_state["profile"]["skills"][:8], separators=(",", ":"))}

Create one practical hands-on task.
"""

        raw_task = await run_adk_agent(
            practice_agent,
            prompt,
            output_key="practice_task",
            user_prefix="practice",
            estimated_output_tokens=300,
            budget="fast",
        )

        task = PracticeTask.model_validate(
            raw_task
        )

        ui_state["current_task"] = task.model_dump()

        return (
            format_task(task),
            ui_state,
        )

    except Exception as exc:
        return (
            f"❌ **Practice generation failed**\n\n`{type(exc).__name__}: {exc}`",
            ui_state,
        )


async def ui_assess_and_adapt(
    learner_submission,
    ui_state,
):
    try:

        if not ui_state:
            return (
                "⚠️ Analyze your profile first.",
                "",
                "",
                ui_state,
            )

        if not ui_state.get(
            "current_task"
        ):
            return (
                "⚠️ Generate a practice task first.",
                "",
                "",
                ui_state,
            )

        if not learner_submission.strip():
            return (
                "⚠️ Submit some work first.",
                "",
                "",
                ui_state,
            )

        task = PracticeTask.model_validate(
            ui_state[
                "current_task"
            ]
        )

        # STAGE 1 — Deep compact analysis.
        prompt_a = f"""
TASK:
{json.dumps(
    task.model_dump(),
    separators=(",", ":")
)}

SUBMISSION:
{learner_submission[:7000]}

Return compact assessment notes only.
"""

        raw_notes = await run_adk_agent(
            assessment_agent_deep,
            prompt_a,
            output_key="assessment_notes",
            user_prefix="deep_notes",
        )

        notes = AssessmentNotes.model_validate(
            raw_notes
        )

        # STAGE 2 — Feed only compact notes to 120B.
        prompt_b = f"""
COMPACT NOTES:
{json.dumps(
    notes.model_dump(),
    separators=(",", ":")
)}

Return the final assessment only.
"""

        try:

            raw_final = await run_adk_agent(
                assessment_final_agent,
                prompt_b,
                output_key="assessment_result",
                user_prefix="deep_final",
            )

        except Exception as second_stage_error:

            # No third LLM call.
            # Convert validated notes directly into the schema.
            if is_rate_limit_error(
                second_stage_error
            ):

                raw_final = {
                    "skill": notes.skill,
                    "score": notes.score_estimate,
                    "strengths": notes.strengths,
                    "weaknesses": notes.weaknesses,
                    "evidence": notes.evidence,
                    "feedback": notes.key_reason,
                    "recommended_action": (
                        "Address the main weakness and resubmit evidence."
                    ),
                }

            else:

                raise

        assessment = AssessmentResult.model_validate(
            raw_final
        )

        skill_name = normalize_skill_name(
            assessment.skill
        )

        if skill_name not in ui_state[
            "skills"
        ]:

            ui_state[
                "skills"
            ][skill_name] = {
                "mastery": 0.0,
                "confidence": 0.3,
                "evidence": [],
                "assessment_scores": [],
            }

        record = ui_state[
            "skills"
        ][skill_name]

        old_mastery = float(
            record["mastery"]
        )

        alpha = 0.30

        record["mastery"] = round(
            max(
                0.0,
                min(
                    1.0,
                    (1 - alpha)
                    * old_mastery
                    + alpha
                    * assessment.score,
                ),
            ),
            3,
        )

        record[
            "assessment_scores"
        ].append(
            assessment.score
        )

        record[
            "evidence"
        ].extend(
            assessment.evidence
        )

        ui_state[
            "latest_assessment"
        ] = assessment.model_dump()

        ui_state[
            "completed_tasks"
        ].append({
            "title": task.title,
            "skill": task.skill,
            "score": assessment.score,
        })

        target_skills = ROLE_SKILLS[
            ui_state[
                "target_role"
            ]
        ]

        new_gaps = []

        for target_skill, target_level in target_skills.items():

            current_record = ui_state[
                "skills"
            ].get(
                target_skill
            )

            if current_record:

                current = current_record[
                    "mastery"
                ]

                confidence = current_record[
                    "confidence"
                ]

            else:

                current = 0.0
                confidence = 0.0

            new_gaps.append({
                "skill": target_skill,
                "current": round(
                    current,
                    2,
                ),
                "target": round(
                    target_level,
                    2,
                ),
                "gap": round(
                    max(
                        target_level
                        - current,
                        0.0,
                    ),
                    2,
                ),
                "confidence": round(
                    confidence,
                    2,
                ),
            })

        new_gaps.sort(
            key=lambda x: x["gap"],
            reverse=True,
        )

        ui_state[
            "gaps"
        ] = new_gaps

        ui_state[
            "plan_version"
        ] += 1

        adaptation = f"""
## 🔄 Learning Path Updated

**{skill_name}**

Before: **{old_mastery:.0%}**  
After: **{record["mastery"]:.0%}**

**Current priority:**  
{new_gaps[0]["skill"] if new_gaps else "None"}

The assessment evidence was incorporated and the remaining gaps were recalculated.
"""

        return (
            format_assessment(
                assessment
            ),
            adaptation,
            "## 🎯 Updated Skill Gaps\n\n"
            + format_gaps(
                new_gaps
            ),
            ui_state,
        )

    except Exception as exc:

        return (
            f"❌ **Assessment failed**\n\n"
            f"`{type(exc).__name__}: {exc}`",
            "",
            "",
            ui_state,
        )


async def ui_replan(ui_state):
    try:
        if not ui_state:
            return "⚠️ Analyze your profile first.", ui_state

        prompt = f"""
TARGET ROLE:
{ui_state["target_role"]}

WEEKLY HOURS:
{ui_state["weekly_hours"]}

CURRENT PLAN VERSION:
{ui_state["plan_version"]}

CURRENT SKILLS:
{json.dumps(list(ui_state["skills"].items())[:6], separators=(",", ":"))}

UPDATED GAPS:
{json.dumps(ui_state["gaps"][:6], separators=(",", ":"))}

LATEST ASSESSMENT:
{json.dumps(ui_state.get("latest_assessment", {}), indent=2)}

PREVIOUS PLAN:
{json.dumps(ui_state.get("learning_plan", {}), indent=2)}

Create a new four-week plan that adapts to the newest evidence.
Do not unnecessarily repeat demonstrated material.
"""

        raw_plan = await run_adk_agent(
            planner_agent,
            prompt,
            output_key="learning_plan",
            user_prefix="replan",
            estimated_output_tokens=420,
            budget="fast",
        )

        plan = LearningPlan.model_validate(
            raw_plan
        )

        ui_state["learning_plan"] = plan.model_dump()

        return (
            f"## 🔄 Updated Learning Plan · v{ui_state['plan_version']}\n\n"
            + format_plan(plan),
            ui_state,
        )

    except Exception as exc:
        return (
            f"❌ **Replanning failed**\n\n`{type(exc).__name__}: {exc}`",
            ui_state,
        )


async def ui_research_resources(ui_state):
    try:
        if not ui_state:
            return "⚠️ Analyze your profile first.", ui_state

        gaps = ui_state.get("gaps", [])

        if not gaps:
            return "No remaining gaps.", ui_state

        priority = gaps[0]

        query = (
            f'{priority["skill"]} tutorial course documentation '
            f'for {ui_state["target_role"]}'
        )

        # Lightweight web tool, no LLM needed for search.
        search_results = await asyncio.to_thread(
            web_search,
            query,
            6,
        )

        if not search_results:
            return (
                "⚠️ No web results were returned. Try again.",
                ui_state,
            )

        curator_prompt = f"""
TARGET SKILL:
{priority["skill"]}

TARGET ROLE:
{ui_state["target_role"]}

LEARNER LEVEL:
Current={priority["current"]}, Target={priority["target"]}

WEB SEARCH RESULTS:
{json.dumps(search_results, indent=2)}

Select 3-4 high-quality learning resources.

Use ONLY URLs contained in the search results.
Do not invent or modify URLs.
Prefer:
- official documentation
- universities
- established learning platforms

Return only ResourceCollection.
"""

        raw_collection = await run_adk_agent(
            resource_curator_agent,
            curator_prompt,
            output_key="resource_collection",
            user_prefix="curator",
        )

        collection = ResourceCollection.model_validate(
            raw_collection
        )

        ui_state["recommended_resources"] = [
            r.model_dump()
            for r in collection.resources
        ]

        return (
            format_resources(collection),
            ui_state,
        )

    except Exception as exc:
        return (
            f"❌ **Research failed**\n\n`{type(exc).__name__}: {exc}`",
            ui_state,
        )


async def ui_progress(ui_state):
    try:
        if not ui_state:
            return "⚠️ Analyze your profile first.", ui_state

        prompt = f"""
TARGET ROLE:
{ui_state["target_role"]}

PLAN VERSION:
{ui_state["plan_version"]}

CURRENT SKILLS:
{json.dumps(list(ui_state["skills"].items())[:6], separators=(",", ":"))}

COMPLETED TASKS:
{json.dumps(ui_state["completed_tasks"], indent=2)}

LATEST ASSESSMENT:
{json.dumps(ui_state.get("latest_assessment", {}), indent=2)}

REMAINING GAPS:
{json.dumps(ui_state["gaps"][:6], separators=(",", ":"))}

Generate a concise progress report.
"""

        raw_report = await run_adk_agent(
            progress_agent,
            prompt,
            output_key="progress_report",
            user_prefix="progress",
        )

        report = ProgressReport.model_validate(
            raw_report
        )

        ui_state["progress_report"] = (
            report.model_dump()
        )

        return (
            format_progress(report),
            ui_state,
        )

    except Exception as exc:
        return (
            f"❌ **Progress report failed**\n\n`{type(exc).__name__}: {exc}`",
            ui_state,
        )


async def ui_ask_edupath(
    question,
    ui_state,
):
    try:
        if not ui_state:
            return "⚠️ Analyze your profile first."

        if not question.strip():
            return "⚠️ Enter a question."

        context = {
            "target_role": ui_state["target_role"],
            "weekly_hours": ui_state["weekly_hours"],
            "skills": ui_state["skills"],
            "gaps": ui_state["gaps"],
            "learning_plan": ui_state.get("learning_plan"),
            "current_task": ui_state.get("current_task"),
            "latest_assessment": ui_state.get(
                "latest_assessment"
            ),
        }

        prompt = f"""
CURRENT EDUPATH STATE:
{json.dumps(context, indent=2)}

LEARNER QUESTION:
{question}
"""

        return await run_adk_agent(
            qa_agent,
            prompt,
            output_key=None,
            user_prefix="qa",
            estimated_output_tokens=180,
            budget="fast",
        )

    except Exception as exc:
        return (
            f"❌ **EduPath could not answer**\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )


# ============================================================
# UI
# ============================================================

CUSTOM_CSS = r"""
:root {
    --bg: #f5f7fb;
    --panel: #ffffff;
    --border: #dbe3ee;
    --text: #172033;
    --muted: #607089;
    --purple: #5b43d6;
    --purple-dark: #4630b0;
    --teal: #087f72;
    --red: #b5345b;
}

body,
.gradio-container {
    background:
        radial-gradient(circle at 95% 0%, rgba(91,67,214,.07), transparent 22%),
        linear-gradient(180deg, #ffffff 0%, var(--bg) 100%) !important;
    color: var(--text) !important;
}

.gradio-container {
    max-width: 1280px !important;
}

.ep-header {
    padding: 20px 4px 10px;
}

.ep-brand {
    font-size: 36px;
    font-weight: 850;
    letter-spacing: -.045em;
    color: #111827;
}

.ep-sub {
    color: var(--muted);
    font-size: 14px;
    font-weight: 650;
    margin-left: 9px;
}

.ep-tagline {
    color: #526179;
    margin-top: 7px;
    font-size: 14px;
}

.ep-pill {
    display: inline-block;
    margin-top: 12px;
    padding: 6px 11px;
    border-radius: 999px;
    color: var(--teal);
    background: #eaf9f6;
    border: 1px solid #c1ebe3;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .04em;
}

.ep-card {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 18px !important;
    box-shadow: 0 8px 24px rgba(26,43,73,.06) !important;
}

.ep-note {
    background: #f8fafc !important;
    border: 1px solid #e3e8ef !important;
    border-radius: 12px !important;
    color: #56657a !important;
    padding: 12px !important;
}

button {
    border-radius: 11px !important;
    font-weight: 700 !important;
}

button.primary {
    background: linear-gradient(135deg, var(--purple), var(--purple-dark)) !important;
    color: white !important;
    border: 0 !important;
    box-shadow: 0 7px 18px rgba(91,67,214,.20) !important;
}

button.primary:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
}

textarea,
input,
select {
    background: #ffffff !important;
    color: var(--text) !important;
    border-color: #c8d3e1 !important;
}

textarea:focus,
input:focus,
select:focus {
    border-color: var(--purple) !important;
    box-shadow: 0 0 0 2px rgba(91,67,214,.10) !important;
}

label,
label span {
    color: #344054 !important;
}

.tab-nav button {
    color: #67768d !important;
    font-weight: 750 !important;
}

.tab-nav button.selected {
    color: #3e2cad !important;
    border-bottom-color: var(--purple) !important;
}

.prose,
.markdown-body {
    color: var(--text) !important;
}

footer {
    display: none !important;
}

.ep-footer {
    color: #8793a6 !important;
    font-size: 11px;
    text-align: center;
    padding: 18px 0 6px;
}
"""


with gr.Blocks(
    title="EduPath — Adaptive Learning Companion",
) as demo:

    ui_state = gr.State(None)

    gr.HTML(
        """
        <div class="ep-header">
            <div class="ep-brand">
                🎓 EduPath
                <span class="ep-sub">Adaptive Learning Companion</span>
            </div>
            <div class="ep-tagline">
                A learning path that changes when your evidence changes.
            </div>
            <div class="ep-pill">● LIVE ADAPTIVE JOURNEY · HYBRID AI</div>
        </div>
        """
    )

    with gr.Tabs():

        # ----------------------------------------------------
        # PROFILE
        # ----------------------------------------------------

        with gr.Tab("01 · Profile & Gaps"):

            with gr.Row():

                with gr.Column(
                    scale=1,
                    elem_classes="ep-card",
                ):

                    gr.Markdown(
                        "### 👤 Tell EduPath about yourself"
                    )

                    target_role = gr.Dropdown(
                        choices=list(ROLE_SKILLS.keys()),
                        value="AI-enabled EV Controls Engineer",
                        label="Target role",
                    )

                    with gr.Row():

                        experience_years = gr.Number(
                            value=4,
                            minimum=0,
                            maximum=40,
                            step=0.5,
                            label="Experience (years)",
                        )

                        weekly_hours = gr.Number(
                            value=8,
                            minimum=1,
                            maximum=40,
                            step=1,
                            label="Learning hours / week",
                        )

                    learner_background = gr.Textbox(
                        label="Background, skills, projects & goals",
                        value=(
                            "I am a power electronics engineer with 4 years of "
                            "automotive/EV experience. I work with PMSM, FOC, "
                            "SVPWM, MATLAB and Simulink. I have some Python "
                            "experience and basic machine-learning knowledge. "
                            "I am learning LLMs, Agentic AI and MCP."
                        ),
                        lines=9,
                    )

                    resume_file = gr.File(
                        label="Optional resume / portfolio",
                        file_types=[".pdf", ".txt"],
                        type="filepath",
                    )

                    analyze_button = gr.Button(
                        "🚀 Analyze My Profile",
                        variant="primary",
                    )

                with gr.Column(
                    scale=1,
                    elem_classes="ep-card",
                ):

                    gr.Markdown(
                        "### 🧭 Your starting point"
                    )

                    profile_output = gr.Markdown(
                        value="Your learner profile will appear here."
                    )

                    gr.Markdown("### 🎯 Skill Gap Map")

                    gap_output = gr.Markdown(
                        value="Your personalized gaps will appear here."
                    )

            analyze_button.click(
                fn=ui_analyze_profile,
                inputs=[
                    target_role,
                    experience_years,
                    weekly_hours,
                    learner_background,
                    resume_file,
                    ui_state,
                ],
                outputs=[
                    profile_output,
                    gap_output,
                    ui_state,
                ],
            )

        # ----------------------------------------------------
        # PLAN
        # ----------------------------------------------------

        with gr.Tab("02 · Learning Plan"):

            with gr.Column(
                elem_classes="ep-card"
            ):

                gr.Markdown(
                    "### 📚 Your personalized learning path"
                )

                gr.Markdown(
                    "Start with **Profile & Gaps**, then generate the plan."
                )

                plan_button = gr.Button(
                    "✨ Generate My Learning Plan",
                    variant="primary",
                )

                plan_output = gr.Markdown()

            plan_button.click(
                fn=ui_generate_plan,
                inputs=[ui_state],
                outputs=[plan_output, ui_state],
            )

        # ----------------------------------------------------
        # PRACTICE / ASSESSMENT
        # ----------------------------------------------------

        with gr.Tab("03 · Practice & Assessment"):

            with gr.Column(
                elem_classes="ep-card"
            ):

                practice_button = gr.Button(
                    "🛠️ Generate Practice Task",
                    variant="primary",
                )

                practice_output = gr.Markdown()

                gr.Markdown("### ✍️ Submit your work")

                learner_submission = gr.Textbox(
                    label="Your answer / work",
                    placeholder=(
                        "Paste your explanation, code summary, "
                        "results or reflection..."
                    ),
                    lines=9,
                )

                assess_button = gr.Button(
                    "🧠 Assess & Adapt My Journey",
                    variant="primary",
                )

                assessment_output = gr.Markdown()
                adaptation_output = gr.Markdown()
                updated_gap_output = gr.Markdown()

                replan_button = gr.Button(
                    "🔄 Generate Updated Plan"
                )

                replan_output = gr.Markdown()

            practice_button.click(
                fn=ui_generate_practice,
                inputs=[ui_state],
                outputs=[
                    practice_output,
                    ui_state,
                ],
            )

            assess_button.click(
                fn=ui_assess_and_adapt,
                inputs=[
                    learner_submission,
                    ui_state,
                ],
                outputs=[
                    assessment_output,
                    adaptation_output,
                    updated_gap_output,
                    ui_state,
                ],
            )

            replan_button.click(
                fn=ui_replan,
                inputs=[ui_state],
                outputs=[
                    replan_output,
                    ui_state,
                ],
            )

        # ----------------------------------------------------
        # RESOURCES / PROGRESS
        # ----------------------------------------------------

        with gr.Tab("04 · Resources & Progress"):

            with gr.Row():

                with gr.Column(
                    elem_classes="ep-card"
                ):

                    gr.Markdown(
                        "### 🔎 Research your priority gap"
                    )

                    research_button = gr.Button(
                        "🌐 Find Learning Resources",
                        variant="primary",
                    )

                    resources_output = gr.Markdown()

                with gr.Column(
                    elem_classes="ep-card"
                ):

                    gr.Markdown(
                        "### 📈 See your journey evolving"
                    )

                    progress_button = gr.Button(
                        "📊 Generate Progress Report",
                        variant="primary",
                    )

                    progress_output = gr.Markdown()

            research_button.click(
                fn=ui_research_resources,
                inputs=[ui_state],
                outputs=[
                    resources_output,
                    ui_state,
                ],
            )

            progress_button.click(
                fn=ui_progress,
                inputs=[ui_state],
                outputs=[
                    progress_output,
                    ui_state,
                ],
            )

        # ----------------------------------------------------
        # ASK
        # ----------------------------------------------------

        with gr.Tab("05 · Ask EduPath"):

            with gr.Column(
                elem_classes="ep-card"
            ):

                gr.Markdown(
                    "### 💬 Ask about your current learning journey"
                )

                question_input = gr.Textbox(
                    label="Your question",
                    placeholder=(
                        "Why is Control Systems my biggest gap? "
                        "What should I focus on next?"
                    ),
                    lines=4,
                )

                ask_button = gr.Button(
                    "💬 Ask EduPath",
                    variant="primary",
                )

                answer_output = gr.Markdown()

            ask_button.click(
                fn=ui_ask_edupath,
                inputs=[
                    question_input,
                    ui_state,
                ],
                outputs=[
                    answer_output,
                ],
            )

    gr.HTML(
        """
        <div class="ep-footer">
            EduPath · Google ADK + Qwen 3.8 27B + GPT-OSS 120B + web search
        </div>
        """
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))

    print(
        f"🚀 Starting EduPath on 0.0.0.0:{port}",
        flush=True,
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        ssr_mode=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
