import dspy
import os
import uuid
from dsp.modules.anthropic import Claude
from dotenv import load_dotenv
from langfuse import Langfuse
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

langfuse = Langfuse()


def create_generation(trace, name, trace_id):
    return trace.generation(name=name, trace_id=trace_id)


def generation_end(generation, output, llm):
    generation.end(
        output=output,
        input={
            "input": llm.history[-1]["kwargs"]["messages"],
            "prompt": llm.history[-1]["prompt"],
        },
        usage={
            "input": llm.history[-1]["response"].usage.input_tokens,
            "output": llm.history[-1]["response"].usage.output_tokens,
        },
        model=llm.history[-1]["kwargs"]["model"],
    )


# Configure dspy
# Sonnet -> claude-3-sonnet-20240229
# Haiku -> claude-3-haiku-20240307
# Opus -> claude-3-opus-20240229
llm = dspy.Anthropic.Claude(
    model="claude-3-haiku-20240307",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=4000,
)

dspy.settings.configure(lm=llm)


# Define the base classes
class JobSkills(BaseModel):
    skill_type: Literal["SOFT SKILL", "TECHNICAL", "BUSINESS"]
    skill_name: str
    referece: str = Field(
        description="Sentence or paragraph where this skill was found in the text"
    )


class BusinessMission(BaseModel):
    company_goal: str = Field(
        description="Goal that is either explicitly state or inferred in the text."
    )
    goal_reference: str = Field(
        description="Sentence(s) or paragraph(s) where the goal is referenced from."
    )


class SkillSignature(dspy.Signature):
    """Extract skills from job description"""

    job_description: str = dspy.InputField(desc="Job description that.")
    job_skills: list[JobSkills] = dspy.OutputField()


class BusinessMissionSignature(dspy.Signature):
    """Extract or infer the Goal of the company"""

    job_description: str = dspy.InputField()
    business_mission: BusinessMission = dspy.OutputField()


class JobInfo(dspy.Module):
    def __init__(self):
        super().__init__()
        self.job_skills = dspy.TypedPredictor(SkillSignature)
        self.business_mission = dspy.TypedPredictor(BusinessMissionSignature)

    def forward(self, job_description, llm, trace, id):
        skills_generation = create_generation(trace=trace, name="skills", trace_id=id)
        skills = self.job_skills(job_description=job_description).job_skills
        generation_end(skills_generation, skills, llm)
        business_mission_generation = create_generation(
            trace=trace, name="business_mission", trace_id=id
        )
        mission = self.business_mission(
            job_description=job_description
        ).business_mission
        generation_end(business_mission_generation, mission, llm)


if __name__ == "__main__":
    job_text = open("job_text.txt", "r").read()
    trace = langfuse.trace(
        name="JobInfo_dspy",
        id=id,
        input={"job_posting": job_text},
    )
    id = str(uuid.uuid4())
    job_info = JobInfo()
    output = job_info(job_description=job_text, llm=llm, trace=trace, id=id)
