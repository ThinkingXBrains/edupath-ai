import os
import json
import uuid
import re
from pathlib import Path

import gradio as gr
from pydantic import BaseModel, Field
from groq import Groq
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "edupath"
MODEL_NAME = "groq/openai/gpt-oss-120b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it as a Hugging Face Space Secret."
    )


def groq_model():
    """Shared ADK/LiteLLM model configuration."""
    return LiteLlm(
        model=MODEL_NAME,
        api_key=GROQ_API_KEY,
        reasoning_effort="low",
        include_reasoning=False,
    )


groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# TARGET ROLE KNOWLEDGE
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
    "pmsm motor": "PMSM",
    "pmsm motors": "PMSM",
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
    "ml": "Machine Learning",
    "llm": "LLMs",
    "llms": "LLMs",
    "large language model": "LLMs",
    "large language models": "LLMs",
    "agentic ai": "Agentic AI",
    "mcp": "MCP",
    "data analysis": "Data Analysis",
    "data analytics": "Data Analysis",
    "python": "Python",
    "python programming": "Python",
    "deep learning": "Deep Learning",
    "matlab": "MATLAB",
    "simulink": "Simulink",
    "apis": "APIs",
    "torque control": "Torque control",
    "regenerative braking": "Regenerative braking",
    "bldc": "BLDC",
    "bldc motor": "BLDC",
    "bldc motors": "BLDC",
}


def normalize_skill_name(name):
    key = " ".join(name.strip().lower().split())
    return SKILL_ALIASES.get(key, name.strip())


def calculate_skill_gaps(profile, target_role):
    target_skills = ROLE_SKILLS[target_role]

    current_skills = {
        normalize_skill_name(skill.name): skill
        for skill in profile.skills
    }

    results = []

    for skill_name, target_level in target_skills.items():
        skill = current_skills.get(skill_name)

        if skill:
            current = skill.proficiency
            confidence = skill.confidence
        else:
            current = 0.0
            confidence = 0.0

        results.append(
            {
                "skill": skill_name,
                "current": round(current, 2),
                "target": round(target_level, 2),
                "gap": round(max(target_level - current, 0.0), 2),
                "confidence": round(confidence, 2),
            }
        )

    results.sort(key=lambda x: x["gap"], reverse=True)
    return results


# ============================================================
# STRUCTURED SCHEMAS
# ============================================================

class Skill(BaseModel):
    name: str
    proficiency: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]


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


class AssessmentResult(BaseModel):
    skill: str
    score: float = Field(ge=0.0, le=1.0)
    strengths: list[str]
    weaknesses: list[str]
    evidence: list[str]
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
# AGENTS
# ============================================================

profile_agent = LlmAgent(
    name="profile_agent",
    model=groq_model(),
    description="Builds a structured learner capability profile.",
    instruction="""
You are EduPath's Profile Analyst.

Analyze the learner information and return a realistic structured learner profile.

Identify:
- target role
- years of experience
- career goals
- available learning time
- existing skills

For every skill:
- proficiency: 0.0 to 1.0
- confidence: 0.0 to 1.0
- evidence: concrete evidence from the input

Rules:
1. Never invent experience or certifications.
2. Never claim expertise without evidence.
3. Distinguish proficiency from confidence.
4. Only include skills supported by the learner information.
5. Use the requested target role, experience and weekly hours when provided.

Return only the structured output.
""",
    output_schema=LearnerProfile,
    output_key="learner_profile",
)


planner_agent = LlmAgent(
    name="planner_agent",
    model=groq_model(),
    description="Creates a personalized learning plan.",
    instruction="""
You are EduPath's Learning Planner.

Create a practical 4-week learning journey from the supplied learner state.

Prioritize meaningful gaps.
Leverage existing strengths.
Respect weekly learning hours.
Avoid reteaching skills already demonstrated strongly.
Include topics, hands-on practice, deliverables and assessments.

Return only the structured learning plan.
""",
    output_schema=LearningPlan,
    output_key="learning_plan",
)


practice_agent = LlmAgent(
    name="practice_agent",
    model=groq_model(),
    description="Creates hands-on practice tasks.",
    instruction="""
You are EduPath's Practice Task Generator.

Create one practical, project-oriented task targeting the learner's highest priority gap.

Match difficulty to the learner's current level.
Leverage domain knowledge when useful.
Produce a concrete deliverable.
Keep the task achievable in the available learning time.

Return only the structured practice task.
""",
    output_schema=PracticeTask,
    output_key="practice_task",
)


assessment_agent = LlmAgent(
    name="assessment_agent",
    model=groq_model(),
    description="Assesses learner submissions using evidence.",
    instruction="""
You are EduPath's Assessment Agent.

Evaluate the learner submission against the practice task.

Assess correctness, completeness, understanding and success criteria.

Rules:
1. Use only evidence in the submission.
2. Do not assume unreported work was completed.
3. Provide a score from 0.0 to 1.0.
4. Identify strengths, weaknesses and evidence.
5. Give constructive feedback.
6. Recommend a concrete next action.

Return only the structured assessment.
""",
    output_schema=AssessmentResult,
    output_key="assessment_result",
)


resource_curator_agent = LlmAgent(
    name="resource_curator_agent",
    model=groq_model(),
    description="Curates verified learning resources.",
    instruction="""
You are EduPath's Resource Curator.

Extract useful learning resources from supplied research.

Rules:
1. Return 3-4 resources when valid resources are available.
2. Use ONLY URLs supplied in the research.
3. Never invent or modify URLs.
4. Match resources to the learner's gap and level.
5. Remove duplicates.

Return only ResourceCollection.
""",
    output_schema=ResourceCollection,
    output_key="resource_collection",
)


progress_agent = LlmAgent(
    name="progress_agent",
    model=groq_model(),
    description="Generates learner progress reports.",
    instruction="""
You are EduPath's Progress Analyst.

Use the learner state and assessment history to produce a concise report.

Classify skills as:
- Acquired: demonstrated strong mastery
- In Progress: meaningful evidence but target not reached

Also identify remaining gaps and actionable next steps.

Do not invent achievements.

Return only ProgressReport.
""",
    output_schema=ProgressReport,
    output_key="progress_report",
)


qa_agent = LlmAgent(
    name="edupath_qa_agent",
    model=groq_model(),
    description="Answers questions about the learner journey.",
    instruction="""
You are EduPath's Learning Companion.

Answer the learner using the supplied current journey context.
Be practical and concise.
Do not invent scores, skills, achievements or resources.
""",
)


# ============================================================
# ADK RUNNER HELPER
# ============================================================

async def run_adk_agent(agent, message_text, output_key=None, user_prefix="edupath"):
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
        parts=[types.Part.from_text(text=message_text)],
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

    text = []

    for event in events:
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    text.append(part.text)

    return "\n".join(text)


# ============================================================
# FORMATTERS
# ============================================================

def format_profile(profile):
    goals = "\n".join(f"- {x}" for x in profile.goals) or "- —"

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
"""


def format_gaps(gaps, limit=None):
    items = gaps[:limit] if limit else gaps

    rows = []
    for item in items:
        current = item["current"]
        target = item["target"]
        filled = int(current * 20)
        bar = "█" * filled + "░" * (20 - filled)

        rows.append(
            f"{item['skill']:25s} {bar} "
            f"{int(current*100):>3}% → {int(target*100):>3}%"
        )

    return "```text\n" + "\n".join(rows) + "\n```"


def format_plan(plan):
    text = f"""
## 📚 Personalized Learning Plan

**{plan.target_role}** · **{plan.total_weeks} weeks** · **{plan.weekly_hours:.1f} hrs/week**
"""

    for week in plan.weeks:
        topics = "\n".join(f"- {x}" for x in week.topics)
        practice = "\n".join(f"- {x}" for x in week.practice_tasks)

        text += f"""
### Week {week.week}
**Objective:** {week.objective}

**Topics**
{topics}

**Practice**
{practice}

**Deliverable:** {week.deliverable}

**Assessment:** {week.assessment}

**Estimated time:** {week.estimated_hours:.1f} hours
"""

    return text


def format_task(task):
    instructions = "\n".join(
        f"{i}. {x}"
        for i, x in enumerate(task.instructions, 1)
    )

    criteria = "\n".join(
        f"- {x}"
        for x in task.success_criteria
    )

    return f"""
## 🛠️ Current Practice

### {task.title}

**Skill:** {task.skill} · **{task.difficulty}** · **{task.estimated_hours:.1f} hrs**

**Objective**  
{task.objective}

**Instructions**
{instructions}

**Deliverable**  
{task.deliverable}

**Success criteria**
{criteria}
"""


def format_assessment(result):
    strengths = "\n".join(f"- {x}" for x in result.strengths)
    weaknesses = "\n".join(f"- {x}" for x in result.weaknesses)

    return f"""
## 📝 Assessment

### {result.score:.0%} — {result.skill}

**Strengths**
{strengths}

**Areas to improve**
{weaknesses}

**Feedback**

{result.feedback}

**Next action**

{result.recommended_action}
"""


def format_resources(collection):
    text = f"""
## 🔎 Recommended Resources

**Current focus:** {collection.skill}
"""

    for i, r in enumerate(collection.resources, 1):
        text += f"""

### {i}. {r.title}

**{r.provider}** · {r.difficulty} · {r.estimated_hours:.1f} hrs · {r.cost}

{r.relevance}

🔗 {r.url}
"""

    return text


def format_progress(report):
    acquired = "\n".join(
        f"- ✅ {x.skill} — {x.mastery:.0%}"
        for x in report.skills_acquired
    ) or "- None yet"

    in_progress = "\n".join(
        f"- 🔄 {x.skill} — {x.mastery:.0%}"
        for x in report.skills_in_progress
    ) or "- None yet"

    gaps = "\n".join(f"- {x}" for x in report.remaining_gaps) or "- None"

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
{gaps}

### Recommended next steps
{next_steps}
"""


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
        if not learner_background.strip() and not resume_file:
            return (
                "⚠️ Add some background or upload a resume.",
                "",
                ui_state,
            )

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

        learner_input = f"""
Target role:
{target_role}

Relevant experience:
{experience_years} years

Available learning time:
{weekly_hours} hours/week

Learner background:
{learner_background}

Resume / portfolio content:
{resume_text[:12000]}
"""

        raw_profile = await run_adk_agent(
            profile_agent,
            learner_input,
            output_key="learner_profile",
            user_prefix="profile",
        )

        profile = LearnerProfile.model_validate(raw_profile)

        profile.target_role = target_role
        profile.experience_years = float(experience_years)
        profile.weekly_hours = float(weekly_hours)

        gaps = calculate_skill_gaps(profile, target_role)

        skills = {}

        for skill in profile.skills:
            name = normalize_skill_name(skill.name)
            skills[name] = {
                "mastery": skill.proficiency,
                "confidence": skill.confidence,
                "evidence": list(skill.evidence),
                "assessment_scores": [],
            }

        state = {
            "target_role": target_role,
            "weekly_hours": float(weekly_hours),
            "profile": profile.model_dump(),
            "skills": skills,
            "gaps": gaps,
            "plan_version": 1,
            "learning_plan": None,
            "current_task": None,
            "completed_tasks": [],
            "latest_assessment": None,
            "recommended_resources": [],
            "progress_report": None,
        }

        return (
            format_profile(profile),
            "## 🎯 Skill Gap Analysis\n\n" + format_gaps(gaps),
            state,
        )

    except Exception as e:
        return (
            f"❌ **Profile analysis failed**\n\n`{type(e).__name__}: {e}`",
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

SKILLS:
{json.dumps(profile["skills"], indent=2)}

SKILL GAPS:
{json.dumps(gaps, indent=2)}

Create a realistic four-week personalized learning plan.
"""

        raw_plan = await run_adk_agent(
            planner_agent,
            prompt,
            output_key="learning_plan",
            user_prefix="planner",
        )

        plan = LearningPlan.model_validate(raw_plan)

        ui_state["learning_plan"] = plan.model_dump()
        ui_state["plan_version"] = max(1, ui_state.get("plan_version", 1))

        return format_plan(plan), ui_state

    except Exception as e:
        return (
            f"❌ **Plan generation failed**\n\n`{type(e).__name__}: {e}`",
            ui_state,
        )


async def ui_generate_practice(ui_state):
    try:
        if not ui_state:
            return "⚠️ Analyze your profile first.", ui_state

        if not ui_state.get("gaps"):
            return "⚠️ No skill gaps are available.", ui_state

        priority = ui_state["gaps"][0]

        prompt = f"""
TARGET ROLE:
{ui_state["target_role"]}

WEEKLY HOURS:
{ui_state["weekly_hours"]}

PRIORITY GAP:
{json.dumps(priority, indent=2)}

CURRENT SKILLS:
{json.dumps(ui_state["profile"]["skills"], indent=2)}

Create one practical hands-on task for this learner.
"""

        raw_task = await run_adk_agent(
            practice_agent,
            prompt,
            output_key="practice_task",
            user_prefix="practice",
        )

        task = PracticeTask.model_validate(raw_task)

        ui_state["current_task"] = task.model_dump()

        return format_task(task), ui_state

    except Exception as e:
        return (
            f"❌ **Practice generation failed**\n\n`{type(e).__name__}: {e}`",
            ui_state,
        )


async def ui_assess_and_adapt(learner_submission, ui_state):
    try:
        if not ui_state:
            return "⚠️ Analyze your profile first.", "", "", ui_state

        if not ui_state.get("current_task"):
            return "⚠️ Generate a practice task first.", "", "", ui_state

        if not learner_submission.strip():
            return "⚠️ Submit your work first.", "", "", ui_state

        task = PracticeTask.model_validate(
            ui_state["current_task"]
        )

        prompt = f"""
PRACTICE TASK:
{json.dumps(task.model_dump(), indent=2)}

LEARNER SUBMISSION:
{learner_submission}

Evaluate the submission using only the available evidence.
"""

        raw = await run_adk_agent(
            assessment_agent,
            prompt,
            output_key="assessment_result",
            user_prefix="assessment",
        )

        assessment = AssessmentResult.model_validate(raw)

        skill_name = normalize_skill_name(assessment.skill)

        if skill_name not in ui_state["skills"]:
            ui_state["skills"][skill_name] = {
                "mastery": 0.0,
                "confidence": 0.3,
                "evidence": [],
                "assessment_scores": [],
            }

        skill = ui_state["skills"][skill_name]
        old_mastery = skill["mastery"]
        alpha = 0.30

        new_mastery = (
            (1 - alpha) * old_mastery
            + alpha * assessment.score
        )

        skill["mastery"] = round(
            max(0.0, min(1.0, new_mastery)), 3
        )

        skill["assessment_scores"].append(
            assessment.score
        )

        skill["evidence"].extend(
            assessment.evidence
        )

        ui_state["completed_tasks"].append(
            {
                "title": task.title,
                "skill": task.skill,
                "score": assessment.score,
                "feedback": assessment.feedback,
            }
        )

        ui_state["latest_assessment"] = (
            assessment.model_dump()
        )

        target_skills = ROLE_SKILLS[
            ui_state["target_role"]
        ]

        new_gaps = []

        for target_skill, target_level in target_skills.items():
            current_skill = ui_state["skills"].get(
                target_skill
            )

            if current_skill:
                current = current_skill["mastery"]
                confidence = current_skill["confidence"]
            else:
                current = 0.0
                confidence = 0.0

            new_gaps.append(
                {
                    "skill": target_skill,
                    "current": round(current, 2),
                    "target": round(target_level, 2),
                    "gap": round(
                        max(target_level - current, 0.0),
                        2,
                    ),
                    "confidence": round(confidence, 2),
                }
            )

        new_gaps.sort(
            key=lambda x: x["gap"],
            reverse=True
        )

        ui_state["gaps"] = new_gaps
        ui_state["plan_version"] += 1

        adaptation = f"""
## 🔄 Learning Path Updated

**{skill_name} mastery**

Before: **{old_mastery:.0%}**  
After: **{skill["mastery"]:.0%}**

**New priority gap:**  
{new_gaps[0]["skill"] if new_gaps else "None"}

The latest assessment evidence has been incorporated
and the learner's remaining gaps have been recalculated.
"""

        return (
            format_assessment(assessment),
            adaptation,
            "## 🎯 Updated Skill Gaps\n\n" + format_gaps(new_gaps),
            ui_state,
        )

    except Exception as e:
        return (
            f"❌ **Assessment failed**\n\n`{type(e).__name__}: {e}`",
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

PLAN VERSION:
{ui_state["plan_version"]}

CURRENT SKILLS:
{json.dumps(ui_state["skills"], indent=2)}

UPDATED GAPS:
{json.dumps(ui_state["gaps"], indent=2)}

LATEST ASSESSMENT:
{json.dumps(ui_state.get("latest_assessment", {}), indent=2)}

PREVIOUS PLAN:
{json.dumps(ui_state.get("learning_plan", {}), indent=2)}

Create a new four-week plan.
Adapt it to the learner's latest demonstrated performance.
Do not repeat demonstrated material unnecessarily.
"""

        raw_plan = await run_adk_agent(
            planner_agent,
            prompt,
            output_key="learning_plan",
            user_prefix="replan",
        )

        plan = LearningPlan.model_validate(raw_plan)

        return (
            f"## 🔄 Updated Learning Plan · v{ui_state['plan_version']}\n\n"
            + format_plan(plan),
            {**ui_state, "learning_plan": plan.model_dump()},
        )

    except Exception as e:
        return (
            f"❌ **Replanning failed**\n\n`{type(e).__name__}: {e}`",
            ui_state,
        )


async def ui_research_resources(ui_state):
    try:
        if not ui_state:
            return "⚠️ Analyze your profile first.", ui_state

        if not ui_state.get("gaps"):
            return "No remaining gaps to research.", ui_state

        priority = ui_state["gaps"][0]

        prompt = f"""
You are EduPath's learning-resource researcher.

Target role:
{ui_state["target_role"]}

Priority skill:
{priority["skill"]}

Current level:
{priority["current"]}

Target level:
{priority["target"]}

Find four high-quality learning resources.

For every resource include:
Title:
Provider:
URL:
Difficulty:
Estimated hours:
Free/Paid:
Why relevant:

URL is mandatory.
Use actual URLs.
Do not invent URLs.
"""

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tools=[
                {
                    "type": "browser_search"
                }
            ],
            tool_choice="required",
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=1500,
        )

        research_text = (
            response.choices[0].message.content or ""
        )

        urls = re.findall(
            r'https?://[^\s\]\)\>,"]+',
            research_text,
        )

        curator_prompt = f"""
TARGET SKILL:
{priority["skill"]}

TARGET ROLE:
{ui_state["target_role"]}

RESEARCH:
{research_text}

VERIFIED URLS:
{json.dumps(urls, indent=2)}

Select 3-4 resources.
Use only the verified URLs above.
Do not invent or alter URLs.
"""

        raw_collection = await run_adk_agent(
            resource_curator_agent,
            curator_prompt,
            output_key="resource_collection",
            user_prefix="resources",
        )

        collection = ResourceCollection.model_validate(
            raw_collection
        )

        ui_state["recommended_resources"] = [
            r.model_dump()
            for r in collection.resources
        ]

        return format_resources(collection), ui_state

    except Exception as e:
        return (
            f"❌ **Research failed**\n\n`{type(e).__name__}: {e}`",
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
{json.dumps(ui_state["skills"], indent=2)}

COMPLETED TASKS:
{json.dumps(ui_state["completed_tasks"], indent=2)}

LATEST ASSESSMENT:
{json.dumps(ui_state.get("latest_assessment", {}), indent=2)}

REMAINING GAPS:
{json.dumps(ui_state["gaps"], indent=2)}

Generate the learner's current progress report.
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

        ui_state["progress_report"] = report.model_dump()

        return format_progress(report), ui_state

    except Exception as e:
        return (
            f"❌ **Progress report failed**\n\n`{type(e).__name__}: {e}`",
            ui_state,
        )


async def ui_ask_edupath(question, ui_state):
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
            "latest_assessment": ui_state.get("latest_assessment"),
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
        )

    except Exception as e:
        return (
            f"❌ **EduPath could not answer**\n\n"
            f"`{type(e).__name__}: {e}`"
        )


# ============================================================
# VISUAL DESIGN
# ============================================================

CUSTOM_CSS = r"""
:root {
    --ep-bg: #0b1020;
    --ep-panel: #121a2c;
    --ep-panel-2: #18233a;
    --ep-border: #263653;
    --ep-text: #edf2ff;
    --ep-muted: #99a8c2;
    --ep-purple: #a177ff;
    --ep-teal: #43dfc2;
    --ep-blue: #6da7ff;
    --ep-pink: #f478b2;
    --ep-danger: #ff718d;
}

body,
.gradio-container {
    background:
        radial-gradient(circle at 85% 5%, rgba(161,119,255,.17), transparent 25%),
        radial-gradient(circle at 5% 92%, rgba(67,223,194,.10), transparent 26%),
        var(--ep-bg) !important;
    color: var(--ep-text) !important;
}

.gradio-container {
    max-width: 1450px !important;
}

.ep-header {
    padding: 24px 4px 18px;
}

.ep-brand {
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -0.04em;
}

.ep-brand-sub {
    color: var(--ep-muted);
    font-size: 14px;
    margin-left: 10px;
}

.ep-tagline {
    color: var(--ep-muted);
    margin-top: 8px;
    font-size: 15px;
}

.ep-status {
    display: inline-block;
    margin-top: 16px;
    border: 1px solid #2a554c;
    background: rgba(17,42,37,.85);
    color: var(--ep-teal);
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 12px;
    letter-spacing: .04em;
}

.ep-section-title {
    margin: 10px 0 8px;
    font-size: 20px;
    font-weight: 750;
}

.ep-card {
    background: linear-gradient(145deg, rgba(18,26,44,.97), rgba(14,21,36,.97));
    border: 1px solid var(--ep-border);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 14px 35px rgba(0,0,0,.18);
}

.ep-card-soft {
    background: rgba(24,35,58,.72);
    border: 1px solid var(--ep-border);
    border-radius: 16px;
    padding: 14px 16px;
}

.ep-metric {
    font-size: 30px;
    font-weight: 800;
}

.ep-muted {
    color: var(--ep-muted);
}

.ep-small {
    font-size: 12px;
    color: var(--ep-muted);
}

button.primary {
    background: linear-gradient(135deg, #a177ff, #7e68ee) !important;
    border: 0 !important;
    box-shadow: 0 8px 22px rgba(126,104,238,.22);
}

button.primary:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
}

button {
    border-radius: 12px !important;
}

textarea,
input,
select {
    background: #0f1728 !important;
    border-color: var(--ep-border) !important;
    color: var(--ep-text) !important;
}

.tabs {
    background: transparent !important;
}

.tab-nav button {
    color: var(--ep-muted) !important;
}

.tab-nav button.selected {
    color: var(--ep-text) !important;
    border-bottom-color: var(--ep-purple) !important;
}

footer {
    display: none !important;
}

.ep-footer {
    color: var(--ep-muted);
    font-size: 11px;
    text-align: center;
    padding: 22px 0 10px;
}

@media (max-width: 900px) {
    .ep-brand {
        font-size: 30px;
    }
}
"""


# ============================================================
# FINAL APP
# ============================================================

with gr.Blocks(
    title="EduPath — Adaptive Learning Companion",
) as demo:

    ui_state = gr.State(None)

    gr.HTML(
        """
        <div class="ep-header">
            <div class="ep-brand">🎓 EduPath <span class="ep-brand-sub">Adaptive Learning Companion</span></div>
            <div class="ep-tagline">
                Build a learning path around what you know, what you need, and how you actually perform.
            </div>
            <div class="ep-status">● LIVE ADAPTIVE JOURNEY</div>
        </div>
        """
    )

    with gr.Tabs():

        # ----------------------------------------------------
        # OVERVIEW
        # ----------------------------------------------------

        with gr.Tab("01 · Profile & Gaps"):

            with gr.Row():

                with gr.Column(
                    scale=1,
                    elem_classes="ep-card"
                ):

                    gr.Markdown(
                        "### 👤 Tell EduPath about yourself"
                    )

                    target_role = gr.Dropdown(
                        choices=list(ROLE_SKILLS.keys()),
                        value="AI-enabled EV Controls Engineer",
                        label="Target Role",
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
                            label="Hours / week",
                        )

                    learner_background = gr.Textbox(
                        label="Background, skills, projects & goals",
                        value=(
                            "I am a power electronics engineer with 4 years of "
                            "automotive and EV experience. I work with PMSM, FOC, "
                            "SVPWM, MATLAB and Simulink. I have some Python experience "
                            "and basic ML knowledge. I am learning LLMs, Agentic AI "
                            "and MCP, and want to combine AI with EV controls."
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
                    elem_classes="ep-card"
                ):

                    gr.Markdown(
                        "### 🧭 Your starting point"
                    )

                    profile_output = gr.Markdown(
                        value=(
                            "Enter your background and click **Analyze My Profile** "
                            "to generate your learner profile."
                        )
                    )

                    gr.Markdown("### 🎯 Skill Gap Map")

                    gap_output = gr.Markdown()

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

            gr.Markdown(
                """
### 📚 Personalized Learning Plan

EduPath uses your current skills and remaining gaps to build the next four weeks of learning.
"""
            )

            plan_button = gr.Button(
                "✨ Generate My Learning Plan",
                variant="primary",
            )

            plan_output = gr.Markdown(
                value="Analyze your profile first."
            )

            plan_button.click(
                fn=ui_generate_plan,
                inputs=[ui_state],
                outputs=[plan_output, ui_state],
            )

        # ----------------------------------------------------
        # PRACTICE / ASSESSMENT
        # ----------------------------------------------------

        with gr.Tab("03 · Practice & Assessment"):

            practice_button = gr.Button(
                "🛠️ Generate Practice Task",
                variant="primary",
            )

            practice_output = gr.Markdown(
                value="Generate a plan first, then create a practice task."
            )

            practice_button.click(
                fn=ui_generate_practice,
                inputs=[ui_state],
                outputs=[practice_output, ui_state],
            )

            gr.Markdown("### ✍️ Submit your work")

            learner_submission = gr.Textbox(
                label="Your answer / work",
                placeholder=(
                    "Paste your explanation, solution, code summary, "
                    "results or reflection here..."
                ),
                lines=10,
            )

            assess_button = gr.Button(
                "📝 Assess & Adapt My Journey",
                variant="primary",
            )

            assessment_output = gr.Markdown()
            adaptation_output = gr.Markdown()
            updated_gap_output = gr.Markdown()

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

            replan_button = gr.Button(
                "🔄 Generate Updated Plan"
            )

            replan_output = gr.Markdown()

            replan_button.click(
                fn=ui_replan,
                inputs=[ui_state],
                outputs=[replan_output, ui_state],
            )

        # ----------------------------------------------------
        # RESOURCES / PROGRESS
        # ----------------------------------------------------

        with gr.Tab("04 · Resources & Progress"):

            with gr.Row():

                with gr.Column(
                    scale=1,
                    elem_classes="ep-card"
                ):

                    gr.Markdown(
                        "### 🔎 Research your current priority gap"
                    )

                    research_button = gr.Button(
                        "🌐 Find Learning Resources",
                        variant="primary",
                    )

                    resources_output = gr.Markdown(
                        value="Your top remaining gap will be researched here."
                    )

                    research_button.click(
                        fn=ui_research_resources,
                        inputs=[ui_state],
                        outputs=[resources_output, ui_state],
                    )

                with gr.Column(
                    scale=1,
                    elem_classes="ep-card"
                ):

                    gr.Markdown(
                        "### 📈 See how your journey is evolving"
                    )

                    progress_button = gr.Button(
                        "📊 Generate Progress Report",
                        variant="primary",
                    )

                    progress_output = gr.Markdown(
                        value="Complete at least one assessment to generate a richer report."
                    )

                    progress_button.click(
                        fn=ui_progress,
                        inputs=[ui_state],
                        outputs=[progress_output, ui_state],
                    )

        # ----------------------------------------------------
        # ASK
        # ----------------------------------------------------

        with gr.Tab("05 · Ask EduPath"):

            gr.Markdown(
                """
### 💬 Your learning companion

Ask why a topic appears in your path, what to focus on next, or how your recent assessment changed the journey.
"""
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
                outputs=[answer_output],
            )

    gr.HTML(
        """
        <div class="ep-footer">
            EduPath · Adaptive Learning Prototype · Powered by Google ADK + Groq
        </div>
        """
    )


if __name__ == "__main__":
    # Render provides PORT at runtime. Render requires the web server
    # to listen on 0.0.0.0:$PORT.
    port = int(os.getenv("PORT", "10000"))

    print(f"🚀 Starting EduPath on 0.0.0.0:{port}", flush=True)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        ssr_mode=False,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
    )
