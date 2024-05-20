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


class RoleValue(BaseModel):
    role_value: str = Field(
        ...,
        description="Value that the role would provide to the value if hired. Explicitly stated or inferred from the Job description.",
    )
    value_reference: str = Field(
        ..., description="Sentence(s) or paragraph(s) where the value was referenced."
    )


class Industry(BaseModel):
    industry: str


class MvpIdea(BaseModel):
    idea: str = Field(
        ...,
        description="Idea for an MVP that the candidate could build based on the job description.",
    )


class JobInformation(BaseModel):
    skills: list[JobSkills]
    mission: BusinessMission
    role_value: RoleValue
    industry: Industry
    mvp: MvpIdea


class SkillSignature(dspy.Signature):
    """Extract skills from job description"""

    job_description: str = dspy.InputField(desc="Job description that.")
    job_skills: list[JobSkills] = dspy.OutputField()


class BusinessMissionSignature(dspy.Signature):
    """Extract or infer the Goal of the company"""

    job_description: str = dspy.InputField()
    business_mission: BusinessMission = dspy.OutputField()


class RoleValueSignature(dspy.Signature):
    """Extract or infer the value the role would provide to the company if hired."""

    job_description: str = dspy.InputField()
    role_value: RoleValue = dspy.OutputField()


class IndustrySignature(dspy.Signature):
    """Infer the industry the business operates in based on the job description."""

    job_description: str = dspy.InputField()
    industry: Industry = dspy.OutputField()


class MvpSignature(dspy.Signature):
    """Generate an MVP idea that a candidate could build based on the job description."""

    job_description: str = dspy.InputField()
    mvp: MvpIdea = dspy.OutputField()


class JobInfo(dspy.Module):
    def __init__(self):
        super().__init__()
        self.job_skills = dspy.TypedPredictor(SkillSignature)
        self.business_mission = dspy.TypedPredictor(BusinessMissionSignature)
        self.role_value = dspy.TypedPredictor(RoleValueSignature)
        self.industry = dspy.TypedPredictor(IndustrySignature)
        self.mvp = dspy.TypedPredictor(MvpSignature)

    def forward(self, job_description):
        skills = self.job_skills(job_description=job_description).job_skills
        mission = self.business_mission(
            job_description=job_description
        ).business_mission
        value = self.role_value(job_description=job_description).role_value
        industry = self.industry(job_description=job_description).industry
        mvp = self.mvp(job_description=job_description).mvp
        return JobInformation(
            skills=skills, mission=mission, role_value=value, industry=industry, mvp=mvp
        )


if __name__ == "__main__":
    job_text = open("job_text.txt", "r").read()
    trace = langfuse.trace(
        name="JobInfo_dspy",
        id=id,
        input={"job_posting": job_text},
    )
    id = str(uuid.uuid4())
    job_info = JobInfo()
    output = job_info(job_description=job_text)
    print(output)
