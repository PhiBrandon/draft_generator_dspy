import dspy
import os
import uuid
from dsp.modules.anthropic import Claude
from dotenv import load_dotenv
from langfuse import Langfuse
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import ast

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
llm = Claude(
    model="claude-3-haiku-20240307",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=4000,
)
llm_2 = Claude(
    model="claude-3-sonnet-20240229",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=4000,
)

dspy.settings.configure(lm=llm)


class JobSkills(BaseModel):
    skill_type: Literal["SOFT SKILL", "TECHNICAL", "BUSINESS"]
    skill_name: str
    referece: str = Field(
        description="Sentence or paragraph where this skill was found in the text"
    )


class SkillSignature(dspy.Signature):
    """Extract skills from job description"""

    job_description: str = dspy.InputField(desc="Job description that.")
    job_skills: list[JobSkills] = dspy.OutputField()


class AssessmentAnswer(BaseModel):
    """Answer to assessment"""

    reasoning: str
    answer: Literal["YES", "NO"] = Field(
        ...,
        description="Yes or No. ALL of the skills need to be present in the job description to be yes.",
    )


class GradeSkills(dspy.Signature):
    """Grade the prescence of skills in a description"""

    assessed_skills = dspy.InputField()
    assessment_job_description = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: AssessmentAnswer = dspy.OutputField()


# Input = Job Description
# Output = List of Job Skills | Type of Skills - Name of Skill - Reference of skill
job_description = open("job_text.txt", "r").read()
df = pd.read_csv("samples_job_1.csv")
skills = df["skills"].to_list()
job_descriptions = df["job_description"].to_list()

skill_pair = dspy.Example(job_description=job_descriptions[0], job_skills=skills[0])

print(skill_pair.job_skills)

trainset = [
    dspy.Example(job_description=job_descriptions[i], job_skills=skills[i])
    for i in range(len(job_descriptions))
]


contains = "Are all of the assessed_skills contained within the job description?"

output = dspy.TypedPredictor(GradeSkills)(
    assessed_skills=trainset[2].job_skills,
    assessment_job_description=trainset[2].job_description,
    assessment_question=contains,
)
output.assessment_answer.answer


# Define metrics to check if skill is in job description
def validate_skills(example, pred, trace=None):
    job_description, job_skills = example.job_description, example.job_skills
    contains = "Are all of the assessed_skills contained within the job description?"
    print(str(pred.job_skills))
    with dspy.context(lm=llm_2):
        contained = dspy.TypedPredictor(GradeSkills)(
            assessed_skills=str(pred.job_skills),
            assessment_job_description=job_description,
            assessment_question=contains,
        )
    print(contained.assessment_answer.answer)
    score = 1 if contained.assessment_answer.answer == "YES" else 0
    return score

overall_score = 0
for a,b in enumerate(trainset):
    pred = dspy.TypedPredictor(SkillSignature)(job_description=b.job_description)
    pred.job_skills
    score = validate_skills(b, pred)
    overall_score+=score

print(f"{overall_score}/{len(trainset)}")
