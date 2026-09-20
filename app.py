import os
import json
import uuid
import asyncio
import time
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import gradio as gr
from pydantic import BaseModel, Field


# ============================================================
# CONFIG — intentionally lightweight at startup
# ============================================================

APP_NAME = "edupath"

FAST_MODEL_NAME = os.getenv(
    "FAST_MODEL_NAME",
    "groq/qwen/qwen3.8-27b",
)

DEEP_MODEL_NAME = os.getenv(
    "DEEP_MODEL_NAME",
    "groq/openai/gpt-oss-120b",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it in Render → Environment."
    )

# These are only used as a conservative client-side output reservation.
# They do NOT bypass Groq limits.
FAST_OUTPUT_BUDGET_PER_MINUTE = 850
DEEP_OUTPUT_BUDGET_PER_MINUTE = 700

_fast_usage: list[tuple[float, int]] = []
_deep_usage: list[tuple[float, int]] = []
_fast_budget_lock: asyncio.Lock | None = None
_deep_budget_lock: asyncio.Lock | None = None


# ============================================================
# LAZY ADK IMPORTS
# Keep Render startup lightweight. ADK/LiteLLM are imported only
# when an actual agent call is made.
# ============================================================

_ADK_CACHE: dict[str, Any] | None = None


def get_adk():
    global _ADK_CACHE

    if _ADK_CACHE is None:
        from google.adk.agents import LlmAgent
        from google.adk.models.lite_llm import LiteLlm
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        _ADK_CACHE = {
            "LlmAgent": LlmAgent,
            "LiteLlm": LiteLlm,
            "InMemoryRunner": InMemoryRunner,
            "types": types,
        }

    return _ADK_CACHE


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

CANONICAL_SKILLS = [
    "Control Systems",
    "FOC",
    "PMSM",
    "SVPWM",
    "MATLAB",
    "Simulink",
    "Power Electronics",
    "Python",
    "Data Analysis",
    "Machine Learning",
    "LLMs",
    "Agentic AI",
    "MCP",
    "Deep Learning",
    "APIs",
    "Torque Control",
    "Regenerative Braking",
    "BLDC",
]


# ============================================================
# SCHEMAS
# ============================================================

class Skill(BaseModel):
    name: str
    proficiency: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


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
    topics: list[str] = Field(max_length=3)
    practice_tasks: list[str] = Field(max_length=1)
    deliverable: str
    assessment: str
    estimated_hours: float


class LearningPlan(BaseModel):
    target_role: str
    total_weeks: int
    weekly_hours: float
    weeks: list[LearningWeek] = Field(max_length=4)


class PracticeTask(BaseModel):
    skill: str
    title: str
    difficulty: str
    objective: str
    instructions: list[str] = Field(max_length=5)
    deliverable: str
    estimated_hours: float
    success_criteria: list[str] = Field(max_length=4)


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


# ============================================================
# SKILL CANONICALIZATION
# ============================================================

SKILL_ALIASES = {
    "power electronics": "Power Electronics",
    "power electronics engineering": "Power Electronics",
    "field oriented control": "FOC",
    "field-oriented control": "FOC",
    "field oriented control (foc)": "FOC",
    "field-oriented control (foc)": "FOC",
    "foc": "FOC",
    "permanent magnet synchronous motor": "PMSM",
    "permanent magnet synchronous motors": "PMSM",
    "permanent magnet synchronous motor (pmsm)": "PMSM",
    "pmsm": "PMSM",
    "space vector pwm": "SVPWM",
    "space vector pwm (svpwm)": "SVPWM",
    "svpwm": "SVPWM",
    "matlab": "MATLAB",
    "simulink": "Simulink",
    "python": "Python",
    "python programming": "Python",
    "machine learning": "Machine Learning",
    "machine learning (basic)": "Machine Learning",
    "basic machine learning": "Machine Learning",
    "data analysis": "Data Analysis",
    "data analytics": "Data Analysis",
    "large language model": "LLMs",
    "large language models": "LLMs",
    "llm": "LLMs",
    "llms": "LLMs",
    "agentic ai": "Agentic AI",
    "mcp": "MCP",
    "deep learning": "Deep Learning",
    "apis": "APIs",
    "control system": "Control Systems",
    "control systems": "Control Systems",
    "torque control": "Torque Control",
    "regenerative braking": "Regenerative Braking",
    "bldc": "BLDC",
    "bldc motor": "BLDC",
    "bldc motors": "BLDC",
}

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
    "power electronics & motor control": ["Power Electronics"],
    "power electronics and motor control": ["Power Electronics"],
}

PHRASE_TO_SKILL = [
    ("field-oriented control", "FOC"),
    ("field oriented control", "FOC"),
    ("space vector pwm", "SVPWM"),
    ("permanent magnet synchronous motor", "PMSM"),
    ("power electronics", "Power Electronics"),
    ("regenerative braking", "Regenerative Braking"),
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


def _norm(value: str) -> str:
    return " ".join(
        value.strip().lower().split()
    )


def normalize_skill_name(name: str) -> str:
    return SKILL_ALIASES.get(
        _norm(name),
        name.strip(),
    )


def expand_skill_name(name: str) -> list[str]:
    key = _norm(name)

    if key in COMPOUND_SKILL_EXPANSIONS:
        return COMPOUND_SKILL_EXPANSIONS[key]

    if key in SKILL_ALIASES:
        return [SKILL_ALIASES[key]]

    found: list[str] = []

    for phrase, canonical in PHRASE_TO_SKILL:
        if phrase in {"foc", "svpwm", "pmsm", "llm", "llms", "mcp", "bldc", "apis"}:
            if re.search(
                rf"\b{re.escape(phrase)}\b",
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
    merged: dict[str, dict[str, Any]] = {}

    for skill in skills:
        for canonical in expand_skill_name(skill.name):
            if canonical not in merged:
                merged[canonical] = {
                    "name": canonical,
                    "proficiency": float(skill.proficiency),
                    "confidence": float(skill.confidence),
                    "evidence": skill.evidence,
                }
            else:
                merged[canonical]["proficiency"] = max(
                    merged[canonical]["proficiency"],
                    float(skill.proficiency),
                )
                merged[canonical]["confidence"] = max(
                    merged[canonical]["confidence"],
                    float(skill.confidence),
                )

    return [
        Skill.model_validate(item)
        for item in merged.values()
    ]


def calculate_skill_gaps(
    profile: LearnerProfile,
    target_role: str,
):
    target_skills = ROLE_SKILLS[target_role]

    current = {
        normalize_skill_name(skill.name): skill
        for skill in profile.skills
    }

    gaps = []

    for skill_name, target in target_skills.items():
        item = current.get(skill_name)

        if item:
            current_value = float(item.proficiency)
            confidence = float(item.confidence)
        else:
            current_value = 0.0
            confidence = 0.0

        gaps.append({
            "skill": skill_name,
            "current": round(current_value, 2),
            "target": round(target, 2),
            "gap": round(
                max(target - current_value, 0.0),
                2,
            ),
            "confidence": round(confidence, 2),
        })

    gaps.sort(
        key=lambda x: x["gap"],
        reverse=True,
    )

    return gaps


# ============================================================
# TOKEN BUDGET
# ============================================================


def _get_lock(kind: str):
    global _fast_budget_lock, _deep_budget_lock

    if kind == "deep":
        if _deep_budget_lock is None:
            _deep_budget_lock = asyncio.Lock()
        return _deep_budget_lock

    if _fast_budget_lock is None:
        _fast_budget_lock = asyncio.Lock()
    return _fast_budget_lock


def _get_bucket(kind: str):
    return _deep_usage if kind == "deep" else _fast_usage


def _get_limit(kind: str):
    return (
        DEEP_OUTPUT_BUDGET_PER_MINUTE
        if kind == "deep"
        else FAST_OUTPUT_BUDGET_PER_MINUTE
    )


async def reserve_output_budget(
    kind: str,
    estimated_tokens: int,
):
    bucket = _get_bucket(kind)
    lock = _get_lock(kind)
    limit = _get_limit(kind)

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
                    (now, estimated_tokens)
                )
                return

            wait_for = (
                bucket[0][0]
                + 60.0
                - now
                + 0.25
            )

        await asyncio.sleep(
            max(wait_for, 0.25)
        )


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in [
            "rate limit",
            "rate_limit",
            "tokens per minute",
            "output tokens per minute",
            "429",
        ]
    )


# ============================================================
# MODEL / AGENT FACTORIES — lazy
# ============================================================


def fast_agent(
    name: str,
    instruction: str,
    output_schema=None,
    output_key: str | None = None,
    max_output_tokens: int = 300,
):
    adk = get_adk()

    kwargs = {
        "name": name,
        "model": adk["LiteLlm"](
            model=FAST_MODEL_NAME,
            api_key=GROQ_API_KEY,
            reasoning_effort="none",
            include_reasoning=False,
        ),
        "description": name,
        "instruction": instruction,
        "generate_content_config": adk["types"].GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=max_output_tokens,
        ),
    }

    if output_schema is not None:
        kwargs["output_schema"] = output_schema

    if output_key:
        kwargs["output_key"] = output_key

    return adk["LlmAgent"](**kwargs)


def deep_text_agent(
    name: str,
    instruction: str,
    max_output_tokens: int = 120,
):
    """
    GPT-OSS 120B agent WITHOUT structured-output validation.

    Assessment intentionally uses compact plain text because the previous
    JSON-schema validation was the failing point.
    """
    adk = get_adk()

    return adk["LlmAgent"](
        name=name,
        model=adk["LiteLlm"](
            model=DEEP_MODEL_NAME,
            api_key=GROQ_API_KEY,
            reasoning_effort="low",
            include_reasoning=False,
        ),
        description=name,
        instruction=instruction,
        generate_content_config=adk["types"].GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        ),
    )


def deep_agent(
    name: str,
    instruction: str,
    output_schema,
    output_key: str,
    max_output_tokens: int,
):
    adk = get_adk()

    return adk["LlmAgent"](
        name=name,
        model=adk["LiteLlm"](
            model=DEEP_MODEL_NAME,
            api_key=GROQ_API_KEY,
            reasoning_effort="low",
            include_reasoning=False,
        ),
        description=name,
        instruction=instruction,
        output_schema=output_schema,
        output_key=output_key,
        generate_content_config=adk["types"].GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        ),
    )


# ============================================================
# RUNNER
# ============================================================

async def run_adk_agent(
    agent,
    message_text: str,
    output_key: str | None,
    user_prefix: str,
    estimated_output_tokens: int,
    budget: str,
):
    await reserve_output_budget(
        budget,
        estimated_output_tokens,
    )

    adk = get_adk()
    user_id = f"{user_prefix}_{uuid.uuid4().hex[:8]}"

    runner = adk["InMemoryRunner"](
        agent=agent,
        app_name=APP_NAME,
    )

    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
    )

    message = adk["types"].Content(
        role="user",
        parts=[
            adk["types"].Part.from_text(
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

        result = state.state.get(
            output_key
        )

        if isinstance(result, str):
            result = json.loads(result)

        return result

    texts = []

    for event in events:
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    texts.append(part.text)

    return "\n".join(texts)



# ============================================================
# COMPACT ASSESSMENT TEXT PARSERS
# ============================================================

def _clean_field(value: str) -> str:
    value = value.strip()
    value = re.sub(
        r"^\s*[-•*]+\s*",
        "",
        value,
    )
    return value[:240].strip()


def _split_items(value: str, max_items: int = 2) -> list[str]:
    if not value:
        return []

    parts = re.split(
        r"\s*(?:\||;|\n)\s*",
        value.strip(),
    )

    clean = []

    for part in parts:
        item = _clean_field(part)
        if item and item.lower() not in {"none", "n/a", "-"}:
            clean.append(item)

        if len(clean) >= max_items:
            break

    return clean


def parse_compact_assessment(
    text: str,
    default_skill: str,
) -> dict:
    """
    Parse the deliberately tiny plain-text assessment format.

    Expected examples:
      SKILL: Control Systems
      SCORE: 0.72
      STRENGTHS: Correct PID structure | Clear explanation
      WEAKNESSES: Missing saturation analysis
      EVIDENCE: Correct transfer function
      REASON: Main concept is understood.

    This parser is intentionally tolerant of Markdown and small wording
    variations so the assessment path does not depend on JSON-schema
    generation.
    """

    if not text:
        raise ValueError(
            "The assessment model returned no text."
        )

    normalized = text.replace(
        "\r",
        "",
    ).strip()

    def capture(label: str) -> str:
        pattern = (
            rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$"
        )

        match = re.search(
            pattern,
            normalized,
        )

        return (
            match.group(1).strip()
            if match
            else ""
        )

    skill = capture("SKILL")
    if not skill:
        skill = default_skill

    score_raw = capture("SCORE")

    score_match = re.search(
        r"(?:0(?:\.\d+)?|1(?:\.0+)?)",
        score_raw,
    )

    if score_match:
        score = float(
            score_match.group(0)
        )
    else:
        # Some models may output "72%".
        percent_match = re.search(
            r"(\d{1,3})\s*%",
            score_raw,
        )

        if percent_match:
            score = (
                float(percent_match.group(1))
                / 100.0
            )
        else:
            score = 0.5

    score = max(
        0.0,
        min(
            1.0,
            score,
        ),
    )

    strengths = _split_items(
        capture("STRENGTHS"),
        max_items=2,
    )

    if not strengths:
        strengths = _split_items(
            capture("STRENGTH"),
            max_items=2,
        )

    weaknesses = _split_items(
        capture("WEAKNESSES"),
        max_items=2,
    )

    if not weaknesses:
        weaknesses = _split_items(
            capture("WEAKNESS"),
            max_items=2,
        )

    evidence = _split_items(
        capture("EVIDENCE"),
        max_items=2,
    )

    reason = (
        capture("REASON")
        or capture("FEEDBACK")
        or "Assessment completed from the submitted evidence."
    )

    action = (
        capture("ACTION")
        or capture("RECOMMENDED_ACTION")
        or "Address the main weakness and resubmit evidence."
    )

    return {
        "skill": skill.strip()[:120],
        "score": round(score, 3),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "evidence": evidence,
        "feedback": _clean_field(reason),
        "recommended_action": _clean_field(action),
    }


# ============================================================
# FORMATTERS
# ============================================================


def format_profile(profile: LearnerProfile):
    goals = "\n".join(
        f"- {goal}"
        for goal in profile.goals
    ) or "- —"

    skills = "\n".join(
        f"- ✅ **{skill.name}** — "
        f"{skill.proficiency:.0%} "
        f"(confidence {skill.confidence:.0%})"
        for skill in profile.skills
    ) or "- No evidence-backed skills detected."

    return f"""
## 👤 Learner Profile

**Target role:** {profile.target_role}

**Experience:** {profile.experience_years:.1f} years

**Learning time:** {profile.weekly_hours:.1f} hrs/week

### Goals
{goals}

### Detected Skills
{skills}
"""


def format_gaps(gaps):
    rows = []

    for item in gaps:
        current = item["current"]
        filled = int(current * 20)
        bar = "█" * filled + "░" * (20 - filled)
        rows.append(
            f"{item['skill']:22s} {bar} "
            f"{int(current*100):>3}% → "
            f"{int(item['target']*100):>3}%"
        )

    return "```text\n" + "\n".join(rows) + "\n```"


def format_plan(plan: LearningPlan):
    out = f"""
## 📚 Personalized Learning Plan

**{plan.target_role}** · **{plan.total_weeks} weeks** · **{plan.weekly_hours:.1f} hrs/week**
"""

    for week in plan.weeks:
        topics = ", ".join(week.topics)
        practice = (
            week.practice_tasks[0]
            if week.practice_tasks
            else "—"
        )

        out += f"""
### Week {week.week}

**Objective:** {week.objective}

**Topics:** {topics}

**Practice:** {practice}

**Deliverable:** {week.deliverable}

**Assessment:** {week.assessment}

**Time:** {week.estimated_hours:.1f} hours
"""

    return out


def format_task(task: PracticeTask):
    instructions = "\n".join(
        f"{i}. {item}"
        for i, item in enumerate(
            task.instructions,
            1,
        )
    )

    criteria = "\n".join(
        f"- {item}"
        for item in task.success_criteria
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


def format_assessment(result: AssessmentResult):
    strengths = "\n".join(
        f"- {item}"
        for item in result.strengths
    ) or "- None recorded"

    weaknesses = "\n".join(
        f"- {item}"
        for item in result.weaknesses
    ) or "- None recorded"

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


def format_resources(items, skill):
    out = f"## 🔎 Recommended Resources\n\n**Current focus:** {skill}\n\n"

    for i, item in enumerate(items, 1):
        title = item.get("title", "Untitled")
        source = item.get("source", item.get("domain", "Web"))
        snippet = item.get("snippet", "").strip()
        url = item.get("url", "")

        out += f"### {i}. {title}\n\n"
        out += f"**{source}**\n"
        if snippet:
            out += f"{snippet}\n"
        if url:
            out += f"🔗 {url}\n"
        out += "\n"

    return out.rstrip()


# ============================================================
# RESEARCH QUALITY CONTROL
# ============================================================

RESEARCH_SOURCES = {
    "Control Systems": ["ocw.mit.edu", "nptel.ac.in", "mathworks.com", "coursera.org"],
    "FOC": ["mathworks.com", "ocw.mit.edu", "nptel.ac.in", "coursera.org"],
    "PMSM": ["mathworks.com", "ocw.mit.edu", "nptel.ac.in", "coursera.org"],
    "SVPWM": ["mathworks.com", "nptel.ac.in", "ocw.mit.edu"],
    "MATLAB": ["mathworks.com", "ocw.mit.edu", "nptel.ac.in", "coursera.org"],
    "Simulink": ["mathworks.com", "ocw.mit.edu", "coursera.org"],
    "Power Electronics": ["ocw.mit.edu", "nptel.ac.in", "mathworks.com", "coursera.org"],
    "Python": ["docs.python.org", "coursera.org", "edx.org", "kaggle.com"],
    "Data Analysis": ["pandas.pydata.org", "coursera.org", "edx.org", "kaggle.com"],
    "Machine Learning": ["scikit-learn.org", "deeplearning.ai", "coursera.org", "edx.org"],
    "LLMs": ["huggingface.co", "learn.microsoft.com", "developers.google.com", "coursera.org"],
    "Agentic AI": ["learn.microsoft.com", "cloud.google.com", "huggingface.co", "coursera.org"],
    "MCP": ["modelcontextprotocol.io", "learn.microsoft.com", "huggingface.co"],
}

RESEARCH_EXCLUDE_DOMAINS = {
    "wikipedia.org", "fandom.com", "youtube.com", "facebook.com",
    "instagram.com", "pinterest.com", "x.com", "twitter.com"
}

RESEARCH_FALLBACKS = {
    "Control Systems": [
        {
            "title": "Feedback Control Systems — MIT OpenCourseWare",
            "source": "MIT OpenCourseWare",
            "url": "https://ocw.mit.edu/courses/16-30-feedback-control-systems-fall-2010/",
            "snippet": "State-space control design, classical analysis, robustness, linearization, and implementation issues.",
        },
        {
            "title": "Control Systems — NPTEL",
            "source": "NPTEL",
            "url": "https://onlinecourses-archive.nptel.ac.in/noc18_ee41/preview",
            "snippet": "Control-system fundamentals covering transfer functions, stability, system response, and PID design.",
        },
        {
            "title": "Understanding Control Systems",
            "source": "MathWorks",
            "url": "https://www.mathworks.com/videos/series/understanding-control-systems-123420.html",
            "snippet": "Introductory feedback and control-system concepts from MathWorks.",
        },
        {
            "title": "Control Systems Courses",
            "source": "Coursera",
            "url": "https://www.coursera.org/courses?query=control%20systems",
            "snippet": "Current control-systems learning options covering dynamics, feedback, PID, and state space.",
        },
    ]
}


def _research_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _research_allowed_domain(url: str, allowed_domains: list[str]) -> bool:
    domain = _research_domain(url)
    if not domain:
        return False
    if any(domain == blocked or domain.endswith("." + blocked) for blocked in RESEARCH_EXCLUDE_DOMAINS):
        return False
    return any(domain == allowed or domain.endswith("." + allowed) for allowed in allowed_domains)


def _research_relevance_score(title: str, snippet: str, skill: str, domain: str) -> int:
    text = f"{title} {snippet}".lower()
    score = 0
    for term in [t for t in re.split(r"[^a-z0-9]+", skill.lower()) if len(t) >= 3]:
        if term in text:
            score += 3
    for term in ["course", "tutorial", "lecture", "documentation", "engineering", "learning", "training", "notes", "control"]:
        if term in text:
            score += 1
    preferred = {
        "ocw.mit.edu": 10, "nptel.ac.in": 9, "mathworks.com": 9,
        "docs.python.org": 10, "pandas.pydata.org": 10, "scikit-learn.org": 10,
        "modelcontextprotocol.io": 10, "learn.microsoft.com": 9, "cloud.google.com": 9,
        "huggingface.co": 8, "deeplearning.ai": 8, "coursera.org": 7,
        "edx.org": 7, "kaggle.com": 6,
    }
    return score + preferred.get(domain, 0)


def _rank_research_results(results: list[dict], skill: str, allowed_domains: list[str], limit: int = 4) -> list[dict]:
    ranked=[]
    seen=set()
    subject_terms=[t for t in re.split(r"[^a-z0-9]+", skill.lower()) if len(t)>=3]
    for item in results:
        url=item.get("href", "")
        title=item.get("title", "").strip()
        snippet=item.get("body", "").strip()
        if not url or not title or url in seen:
            continue
        if not _research_allowed_domain(url, allowed_domains):
            continue
        domain=_research_domain(url)
        text=f"{title} {snippet}".lower()
        if not any(term in text for term in subject_terms):
            continue
        ranked.append({
            "title": title,
            "url": url,
            "snippet": snippet[:450],
            "domain": domain,
            "source": domain,
            "score": _research_relevance_score(title, snippet, skill, domain),
        })
        seen.add(url)
    ranked.sort(key=lambda x:x["score"], reverse=True)
    return ranked[:limit]


async def search_quality_resources(skill: str, target_role: str) -> list[dict]:
    allowed=RESEARCH_SOURCES.get(
        skill,
        ["ocw.mit.edu", "nptel.ac.in", "mathworks.com", "coursera.org"],
    )
    from ddgs import DDGS

    async def one_domain(domain: str):
        query=f'"{skill}" tutorial course site:{domain}'
        try:
            return await asyncio.to_thread(
                lambda: list(
                    DDGS().text(
                        query,
                        max_results=3,
                        region="us-en",
                        safesearch="moderate",
                        backend="auto",
                    )
                )
            )
        except Exception:
            return []

    batches=await asyncio.gather(
        *[one_domain(domain) for domain in allowed[:4]]
    )

    combined=[item for batch in batches for item in batch]
    ranked=_rank_research_results(
        combined, skill, allowed, limit=4
    )

    if ranked:
        return ranked

    return RESEARCH_FALLBACKS.get(skill, [])


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
                resume_text = Path(path).read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

        prompt = f"""
LEARNER BACKGROUND:
{learner_background[:5000]}

RESUME EXCERPT:
{resume_text[:5000]}
"""

        agent = fast_agent(
            name="profile_agent",
            instruction="""
Extract ONLY the learner's goals and evidence-backed skills.

The UI already supplies target role, experience years and weekly learning hours.
Do not output those fields.

Canonical skill names:
Control Systems, FOC, PMSM, SVPWM, MATLAB, Simulink,
Power Electronics, Python, Data Analysis, Machine Learning,
LLMs, Agentic AI, MCP, Deep Learning, APIs,
Torque Control, Regenerative Braking, BLDC.

Rules:
- maximum 6 skills
- maximum 3 goals
- one short evidence sentence per skill
- use simple canonical skill names
- do not invent evidence
- no explanations
- return only structured output
""",
            output_schema=ProfileInsights,
            output_key="profile_insights",
            max_output_tokens=380,
        )

        raw = await run_adk_agent(
            agent,
            prompt,
            output_key="profile_insights",
            user_prefix="profile",
            estimated_output_tokens=380,
            budget="fast",
        )

        insights = ProfileInsights.model_validate(raw)
        skills = canonicalize_profile_skills(
            insights.skills
        )

        profile = LearnerProfile(
            target_role=target_role,
            experience_years=float(experience_years),
            goals=insights.goals,
            weekly_hours=float(weekly_hours),
            skills=skills,
        )

        gaps = calculate_skill_gaps(
            profile,
            target_role,
        )

        state = {
            "target_role": target_role,
            "weekly_hours": float(weekly_hours),
            "profile": profile.model_dump(),
            "skills": {
                skill.name: {
                    "mastery": skill.proficiency,
                    "confidence": skill.confidence,
                    "evidence": [skill.evidence],
                    "assessment_scores": [],
                }
                for skill in skills
            },
            "gaps": gaps,
            "plan_version": 1,
            "learning_plan": None,
            "current_task": None,
            "completed_tasks": [],
            "latest_assessment": None,
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

        prompt = f"""
TARGET ROLE: {profile['target_role']}
HOURS/WEEK: {profile['weekly_hours']}
SKILLS: {json.dumps(profile['skills'][:6], separators=(',', ':'))}
TOP GAPS: {json.dumps(ui_state['gaps'][:6], separators=(',', ':'))}

Create a compact 4-week plan.
"""

        agent = fast_agent(
            name="planner_agent",
            instruction="""
Create a compact four-week learning plan.

Limits:
- exactly 4 weeks
- max 3 topics/week
- exactly 1 practice item/week
- short objective, deliverable and assessment
- prioritize the largest gaps
- leverage demonstrated strengths
- no long explanations
- return only structured output
""",
            output_schema=LearningPlan,
            output_key="learning_plan",
            max_output_tokens=420,
        )

        raw = await run_adk_agent(
            agent,
            prompt,
            output_key="learning_plan",
            user_prefix="planner",
            estimated_output_tokens=420,
            budget="fast",
        )

        plan = LearningPlan.model_validate(raw)
        ui_state["learning_plan"] = plan.model_dump()

        return format_plan(plan), ui_state

    except Exception as exc:
        return (
            f"❌ **Plan generation failed**\n\n"
            f"`{type(exc).__name__}: {exc}`",
            ui_state,
        )


async def ui_generate_practice(ui_state):
    try:
        if not ui_state:
            return "⚠️ Analyze your profile first.", ui_state

        priority = ui_state["gaps"][0]

        prompt = f"""
TARGET ROLE: {ui_state['target_role']}
PRIORITY GAP: {json.dumps(priority, separators=(',', ':'))}
EXISTING SKILLS: {json.dumps(ui_state['profile']['skills'][:6], separators=(',', ':'))}

Create one practical task.
"""

        agent = fast_agent(
            name="practice_agent",
            instruction="""
Create ONE practical hands-on task for the priority skill gap.

Limits:
- max 5 instructions
- max 4 success criteria
- concise sentences
- concrete deliverable
- no long explanation
- return only structured output
""",
            output_schema=PracticeTask,
            output_key="practice_task",
            max_output_tokens=300,
        )

        raw = await run_adk_agent(
            agent,
            prompt,
            output_key="practice_task",
            user_prefix="practice",
            estimated_output_tokens=300,
            budget="fast",
        )

        task = PracticeTask.model_validate(raw)
        ui_state["current_task"] = task.model_dump()

        return format_task(task), ui_state

    except Exception as exc:
        return (
            f"❌ **Practice generation failed**\n\n"
            f"`{type(exc).__name__}: {exc}`",
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

        # ==================================================
        # GPT-OSS 120B — STAGE 1
        # NO JSON SCHEMA.
        # This avoids the exact json_validate_failed error
        # seen in the previous deployment.
        # ==================================================

        notes_agent = deep_text_agent(
            name="assessment_notes_agent",
            instruction="""
You are a technical assessment analyst.

Analyze the learner submission against the assigned task.

Return EXACTLY these six lines and nothing else:

SKILL: <skill>
SCORE: <0.0 to 1.0>
STRENGTHS: <item> | <item>
WEAKNESSES: <item> | <item>
EVIDENCE: <item> | <item>
REASON: <one short sentence>

Rules:
- maximum 2 items after each list label
- keep each item short
- score must be numeric
- use only evidence in the submission
- do not invent missing work
- no markdown
- no JSON
""",
            max_output_tokens=120,
        )

        task_compact = {
            "skill": task.skill,
            "title": task.title,
            "objective": task.objective,
            "success_criteria": task.success_criteria,
        }

        prompt_a = f"""
TASK:
{json.dumps(
    task_compact,
    separators=(",", ":"),
)}

SUBMISSION:
{learner_submission[:6000]}

Return exactly the six requested lines.
"""

        notes_text = await run_adk_agent(
            notes_agent,
            prompt_a,
            output_key=None,
            user_prefix="deep_notes",
            estimated_output_tokens=120,
            budget="deep",
        )

        notes = parse_compact_assessment(
            notes_text,
            default_skill=task.skill,
        )

        # ==================================================
        # GPT-OSS 120B — STAGE 2
        # Only compact notes are sent.
        # Still plain text, so there is NO JSON-schema failure.
        # ==================================================

        final_agent = deep_text_agent(
            name="assessment_final_agent",
            instruction="""
Review the compact assessment notes.

Return EXACTLY these five lines and nothing else:

SCORE: <0.0 to 1.0>
STRENGTHS: <item> | <item>
WEAKNESSES: <item> | <item>
FEEDBACK: <one short sentence>
ACTION: <one short sentence>

Rules:
- preserve the evidence-based score
- maximum 2 items per list
- do not introduce new evidence
- no JSON
- no markdown
""",
            max_output_tokens=110,
        )

        prompt_b = f"""
COMPACT NOTES:
{json.dumps(
    notes,
    separators=(",", ":"),
)}

Return exactly the five requested lines.
"""

        try:

            final_text = await run_adk_agent(
                final_agent,
                prompt_b,
                output_key=None,
                user_prefix="deep_final",
                estimated_output_tokens=110,
                budget="deep",
            )

            refined = parse_compact_assessment(
                final_text,
                default_skill=notes["skill"],
            )

            # Stage 2 may omit evidence by design. Preserve the
            # evidence captured by Stage 1.
            refined["evidence"] = (
                notes["evidence"]
            )

            # Guard against an accidental score jump caused by
            # an unconstrained second-pass rewrite.
            if abs(
                refined["score"]
                - notes["score"]
            ) > 0.25:

                refined["score"] = (
                    notes["score"]
                )

            if not refined["strengths"]:
                refined["strengths"] = (
                    notes["strengths"]
                )

            if not refined["weaknesses"]:
                refined["weaknesses"] = (
                    notes["weaknesses"]
                )

            if not refined["feedback"]:
                refined["feedback"] = (
                    notes["feedback"]
                )

            assessment_data = refined

        except Exception as second_stage_error:

            # Stage 1 is already a complete assessment.
            # Do NOT make a third LLM call.
            assessment_data = notes

        assessment = AssessmentResult.model_validate(
            assessment_data
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

        before = float(
            record["mastery"]
        )

        alpha = 0.30

        after = (
            (1 - alpha)
            * before
            + alpha
            * assessment.score
        )

        record[
            "mastery"
        ] = round(
            max(
                0.0,
                min(
                    1.0,
                    after,
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
        ].append(
            {
                "title": task.title,
                "skill": task.skill,
                "score": assessment.score,
            }
        )

        ui_state[
            "gaps"
        ] = calculate_skill_gaps(
            LearnerProfile.model_validate(
                ui_state[
                    "profile"
                ]
                | {
                    "skills": [
                        {
                            "name": name,
                            "proficiency": value[
                                "mastery"
                            ],
                            "confidence": value[
                                "confidence"
                            ],
                            "evidence": (
                                value["evidence"][0]
                                if value["evidence"]
                                else ""
                            ),
                        }
                        for name, value
                        in ui_state[
                            "skills"
                        ].items()
                    ],
                }
            ),
            ui_state[
                "target_role"
            ],
        )

        ui_state[
            "plan_version"
        ] += 1

        adaptation = f"""
## 🔄 Learning Path Updated

**{skill_name}**

Before: **{before:.0%}**  
After: **{record["mastery"]:.0%}**

**Current priority:**  
{ui_state["gaps"][0]["skill"] if ui_state["gaps"] else "None"}

The assessment evidence was incorporated and the remaining gaps were recalculated.
"""

        return (
            format_assessment(
                assessment
            ),
            adaptation,
            "## 🎯 Updated Skill Gaps\n\n"
            + format_gaps(
                ui_state["gaps"]
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
    return await ui_generate_plan(ui_state)


async def ui_research_resources(ui_state):
    try:
        if not ui_state:
            return "⚠️ Analyze your profile first.", ui_state

        gaps = ui_state.get("gaps", [])
        if not gaps:
            return "No remaining gaps.", ui_state

        skill = gaps[0]["skill"]

        resources = await search_quality_resources(
            skill,
            ui_state["target_role"],
        )

        if not resources:
            return (
                f"⚠️ No high-quality learning resources found for **{skill}** right now.",
                ui_state,
            )

        ui_state["recommended_resources"] = resources

        return (
            format_resources(resources, skill),
            ui_state,
        )

    except Exception as exc:
        return (
            f"❌ **Research failed**\n\n`{type(exc).__name__}: {exc}`",
            ui_state,
        )


async def ui_progress(ui_state):
    if not ui_state:
        return "⚠️ Analyze your profile first.", ui_state

    acquired = []
    in_progress = []
    remaining = []

    targets = ROLE_SKILLS[
        ui_state["target_role"]
    ]

    for skill, target in targets.items():
        record = ui_state["skills"].get(skill)
        mastery = (
            record["mastery"]
            if record
            else 0.0
        )

        if mastery >= 0.85 * target:
            acquired.append(
                f"{skill} — {mastery:.0%}"
            )
        elif mastery > 0:
            in_progress.append(
                f"{skill} — {mastery:.0%}"
            )

        if mastery < target:
            remaining.append(
                f"{skill} ({mastery:.0%} → {target:.0%})"
            )

    top_gap = (
        ui_state["gaps"][0]["skill"]
        if ui_state.get("gaps")
        else "None"
    )

    out = f"""
## 📈 Progress Report

**Plan version:** {ui_state.get('plan_version', 1)}

**Assessed tasks:** {len(ui_state.get('completed_tasks', []))}

### Acquired
"""

    out += (
        "\n".join(
            f"- ✅ {x}" for x in acquired
        )
        or "- None yet"
    )

    out += "\n\n### In Progress\n"
    out += (
        "\n".join(
            f"- 🔄 {x}" for x in in_progress
        )
        or "- None yet"
    )

    out += "\n\n### Remaining Gaps\n"
    out += (
        "\n".join(
            f"- {x}" for x in remaining[:8]
        )
        or "- None"
    )

    out += f"\n\n### Next Step\n\nFocus next on **{top_gap}**."

    return out, ui_state


async def ui_ask_edupath(
    question,
    ui_state,
):
    try:
        if not ui_state:
            return "⚠️ Analyze your profile first."

        if not question.strip():
            return "⚠️ Enter a question."

        compact_state = {
            "target_role": ui_state["target_role"],
            "top_gaps": ui_state["gaps"][:5],
            "skills": {
                name: {
                    "mastery": value["mastery"],
                    "confidence": value["confidence"],
                }
                for name, value
                in list(ui_state["skills"].items())[:6]
            },
            "current_task": ui_state.get(
                "current_task"
            ),
        }

        prompt = f"""
STATE:
{json.dumps(compact_state, separators=(',', ':'))}

QUESTION:
{question[:800]}
"""

        agent = fast_agent(
            name="qa_agent",
            instruction="""
Answer the learner using only the supplied current state.

Maximum 120 words.
Do not invent scores, achievements or skills.
Be practical and direct.
""",
            max_output_tokens=180,
        )

        return await run_adk_agent(
            agent,
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
    --muted: #61708a;
    --purple: #6046d8;
    --purple-dark: #4c35b6;
    --teal: #087f72;
}

body,
.gradio-container {
    background:
        radial-gradient(circle at 92% 0%, rgba(96,70,216,.07), transparent 22%),
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
    color: #111827 !important;
}

.ep-sub {
    color: var(--muted) !important;
    font-size: 14px;
    font-weight: 650;
    margin-left: 9px;
}

.ep-tagline {
    color: #526179 !important;
    margin-top: 7px;
    font-size: 14px;
}

.ep-pill {
    display: inline-block;
    margin-top: 12px;
    padding: 6px 11px;
    border-radius: 999px;
    color: var(--teal) !important;
    background: #e8f9f5;
    border: 1px solid #bfe9df;
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

button {
    border-radius: 11px !important;
    font-weight: 700 !important;
}

button.primary {
    background: linear-gradient(135deg, var(--purple), var(--purple-dark)) !important;
    color: #fff !important;
    border: 0 !important;
    box-shadow: 0 7px 18px rgba(96,70,216,.20) !important;
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
    box-shadow: 0 0 0 2px rgba(96,70,216,.10) !important;
}

label,
label span {
    color: #344054 !important;
}

.tab-nav button {
    color: #64748b !important;
    font-weight: 750 !important;
}

.tab-nav button.selected {
    color: #3c2ca7 !important;
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
    color: #8390a5 !important;
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
            <div class="ep-pill">
                ● LIVE ADAPTIVE JOURNEY · TWO-MODEL AI
            </div>
        </div>
        """
    )

    with gr.Tabs():

        with gr.Tab("01 · Profile & Gaps"):
            with gr.Row():
                with gr.Column(
                    elem_classes="ep-card"
                ):
                    gr.Markdown(
                        "### 👤 Tell EduPath about yourself"
                    )

                    target_role = gr.Dropdown(
                        choices=list(
                            ROLE_SKILLS.keys()
                        ),
                        value=(
                            "AI-enabled EV "
                            "Controls Engineer"
                        ),
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
                            "I am a power electronics engineer with 4 years "
                            "of automotive and EV experience. I work with "
                            "PMSM, FOC, SVPWM, MATLAB and Simulink. I have "
                            "some Python and basic ML knowledge. I am learning "
                            "LLMs, Agentic AI and MCP and want to combine them "
                            "with EV controls."
                        ),
                        lines=9,
                    )

                    resume_file = gr.File(
                        label="Optional resume / portfolio",
                        file_types=[
                            ".pdf",
                            ".txt",
                        ],
                        type="filepath",
                    )

                    analyze_button = gr.Button(
                        "🚀 Analyze My Profile",
                        variant="primary",
                    )

                with gr.Column(
                    elem_classes="ep-card"
                ):
                    gr.Markdown(
                        "### 🧭 Your starting point"
                    )

                    profile_output = gr.Markdown(
                        "Your learner profile will appear here."
                    )

                    gr.Markdown(
                        "### 🎯 Skill Gap Map"
                    )

                    gap_output = gr.Markdown(
                        "Your personalized gaps will appear here."
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

        with gr.Tab("02 · Learning Plan"):
            with gr.Column(
                elem_classes="ep-card"
            ):
                gr.Markdown(
                    "### 📚 Your personalized learning path"
                )

                plan_button = gr.Button(
                    "✨ Generate My Learning Plan",
                    variant="primary",
                )

                plan_output = gr.Markdown()

            plan_button.click(
                fn=ui_generate_plan,
                inputs=[ui_state],
                outputs=[
                    plan_output,
                    ui_state,
                ],
            )

        with gr.Tab("03 · Practice & Assessment"):
            with gr.Column(
                elem_classes="ep-card"
            ):
                practice_button = gr.Button(
                    "🛠️ Generate Practice Task",
                    variant="primary",
                )

                practice_output = gr.Markdown()

                gr.Markdown(
                    "### ✍️ Submit your work"
                )

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
            EduPath · Google ADK · Qwen 3.8 27B + GPT-OSS 120B
        </div>
        """
    )


# ============================================================
# RENDER STARTUP
# ============================================================

if __name__ == "__main__":
    port = int(
        os.getenv(
            "PORT",
            "10000",
        )
    )

    print(
        f"🚀 EduPath startup: binding to 0.0.0.0:{port}",
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
