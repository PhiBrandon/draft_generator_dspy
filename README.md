# Proposal Draft Generator

1. Create virtual environment `python3 -m venv dspy-venv`
2. Activate -  source dspy-venv/bin/activate
3. Install packages - pip install -r requirements.txt
4. Create .env file
    -  Get API key from https://console.anthropic.com/settings/keys
    -  Add ANTHROPIC_API_KEY=your_api_key

5. Update `job_posting.txt` with the job posting you want to write a proposal for.
6. Update `resume.txt` with any relevant text information about your credentials.
7. Run `python start.py`


# Job Information Generator
1. Follow above steps to setup the environment
2. Update the job_text.txt file with the job description.
3. Run `python job_skills.py`