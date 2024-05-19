import dspy
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field


load_dotenv()
# Sonnet -> claude-3-sonnet-20240229
# Haiku -> claude-3-haiku-20240307
# Opus -> claude-3-opus-20240229
llm_2 = dspy.Anthropic.Claude(
    model="claude-3-haiku-20240307",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=4000,
)
dspy.settings.configure(lm=llm_2)

job_posting = open("job_posting.txt", "r").read()
resume = open("resume.txt", "r").read()

proposal_generator = dspy.Predict("job_posting, resume -> proposal")


class Proposal(BaseModel):
    """Proposal in the structure # Title\n #Proposal Content"""

    scratch_pad: str
    proposal: str


class Poc(BaseModel):
    """Proof of concent plan professionally structured"""

    scratch_pad: str
    poc_plan: str = Field(
        ..., description="Proof of concept timeline bulleted task list"
    )


class Mvp(BaseModel):
    """Minimum Viable Product plan professionally structured"""

    scratch_pad: str
    mvp_idea: str = Field(
        ...,
        description="Mvp idea that that is clearly defined based on the job posting.",
    )
    mvp: str = Field(
        ...,
        description="""Minimum Viable Product plan professionally structured. This should contain a timeline for each specific phase, broken down weekly. This should consider the time constraint.""",
    )


class Report(BaseModel):
    """Detailed report professionally structured."""

    scratch_pad: str
    report: str = Field(
        ...,
        description=""" comprehensive and detailed report based on the information from the following components: Proposal, Job Posting, Proof of Concept Plan, MVP Schedule.""",
    )


class Revision(BaseModel):
    """Revisions professionally structured."""

    revised_poc: str = Field(
        ...,
        description="""revision of original POC based on the information in the provided detailed report""",
    )
    revised_proposal: str = Field(
        ...,
        description="""revision of original proposal based on the information in the provided detailed report""",
    )
    revised_mvp: str = Field(
        ...,
        description="""revision of original mvp based on the information in the provided detailed report""",
    )


class FinalDocument(BaseModel):
    """Polished final document, professionally structured."""
    final_document: str = Field(
        ...,
        description=""" Polished final document that only structures the revised proposal, POC plan, and MVP schedule. This will be shared with project stakeholders, so ensure it is well-organized, clearly written, and professionally formatted""",
    )


class Combined(BaseModel):
    proposol: Proposal
    poc: Poc
    mvp: Mvp
    report: Report
    revisions: Revision
    final_document: FinalDocument


class ProposalGenerator(dspy.Signature):
    """Generate a proposal for technical contract work."""

    job_posting: str = dspy.InputField(desc="The posting for a job.")
    resume: str = dspy.InputField(
        desc="The resume of the candidate applying for the job"
    )
    proposal: Proposal = dspy.OutputField()


class PocGen(dspy.Signature):
    """Generate a proof of concept plan"""

    job_posting: str = dspy.InputField(desc="The posting for a job.")
    resume: str = dspy.InputField(
        desc="The resume of the candidate applying for the job"
    )
    proposal: Proposal = dspy.InputField()
    time_to_deliver: str = dspy.InputField()
    poc_plan: Poc = dspy.OutputField()


class MvpGen(dspy.Signature):
    """Generate a Minimum viable product plan"""

    job_posting: str = dspy.InputField(desc="The posting for a job.")
    proposal: Proposal = dspy.InputField()
    time_constraint: str = dspy.InputField()
    poc_plan: Poc = dspy.InputField()
    mvp: Mvp = dspy.OutputField()


class ReportGen(dspy.Signature):
    """Generate a comprehensive and detailed report based on the information from the following components: Proposal, Job Posting, Proof of Concept Plan, MVP Schedule"""

    job_posting: str = dspy.InputField(desc="The posting for a job.")
    proposal: Proposal = dspy.InputField()
    poc_plan: Poc = dspy.InputField()
    mvp: Mvp = dspy.InputField()
    report: Report = dspy.OutputField()


class RevisionGen(dspy.Signature):
    """Generate revisions for the following: Proposal, Proof of Concept Plan, MVP Schedule. Use the information from the detailed report and each of the originals."""

    proposal: Proposal = dspy.InputField()
    poc_plan: Poc = dspy.InputField()
    mvp: Mvp = dspy.InputField()
    report: Report = dspy.InputField()
    revisions: Revision = dspy.OutputField()


class FinalGen(dspy.Signature):
    """Generate Final document only structuring on the following: Revised Proposol, Revised Proof of Concept Plan, Revised MVP Schedule, and Job Posting."""

    job_posting: str = dspy.InputField(desc="The posting for a job.")
    proposal: str = dspy.InputField()
    poc_plan: str = dspy.InputField()
    mvp: str = dspy.InputField()
    final_document: FinalDocument = dspy.OutputField()


def create_generation(trace, name, trace_id):
    return trace.generation(name=name, trace_id=trace_id)


class DataDocGen(dspy.Module):
    def __init__(self):
        super().__init__()
        self.proposal = dspy.TypedPredictor(ProposalGenerator)
        self.poc = dspy.TypedPredictor(PocGen)
        self.mvp = dspy.TypedPredictor(MvpGen)
        self.report = dspy.TypedPredictor(ReportGen)
        self.revision = dspy.TypedPredictor(RevisionGen)
        self.final_document = dspy.TypedPredictor(FinalGen)

    def forward(self, resume, job_posting, time_to_deliver, time_constraint):

        proposal = self.proposal(resume=resume, job_posting=job_posting).proposal
        print(proposal)
        poc = self.poc(
            proposal=proposal,
            job_posting=job_posting,
            resume=resume,
            time_to_deliver=time_to_deliver,
        ).poc_plan
        print(poc)
        mvp = self.mvp(
            job_posting=job_posting,
            proposal=proposal,
            time_constraint=time_constraint,
            poc_plan=poc,
        ).mvp
        print(mvp)
        report = self.report(
            job_posting=job_posting, proposal=proposal, poc_plan=poc, mvp=mvp
        ).report
        print(report)
        revisions = self.revision(
            proposal=proposal, poc_plan=poc, mvp=mvp, report=report
        ).revisions
        print(revisions)
        final_document = self.final_document(
            job_posting=job_posting,
            proposal=revisions.revised_proposal,
            poc_plan=revisions.revised_poc,
            mvp=revisions.revised_mvp,
        ).final_document
        print(final_document)
        combined = Combined(
            proposol=proposal,
            poc=poc,
            mvp=mvp,
            report=report,
            revisions=revisions,
            final_document=final_document,
        )
        return combined


data_doc = DataDocGen()
combined = data_doc(
    job_posting=job_posting,
    resume=resume,
    # Time to deliver the POC
    time_to_deliver="12 hours",
    # Time constraint for MVP development
    time_constraint="20 hours per week 2 months",
)
print(combined)
