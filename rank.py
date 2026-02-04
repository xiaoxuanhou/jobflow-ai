"""
Job ranking module for jobflow-ai.

Performs consistent ranking of all viable jobs in a single API call.
This solves the cross-batch inconsistency problem with independent scoring.
"""

import json
import re
from typing import Optional

import anthropic
import yaml

from budget import BudgetTracker, estimate_cost


def rank_jobs_consistently(
    viable_jobs: list[dict],
    resume: dict,
    criteria: dict,
    budget_tracker: Optional[BudgetTracker] = None
) -> list[dict]:
    """
    Rank all viable jobs in a SINGLE API call for consistency.

    Input: All jobs scoring >= threshold from filter stage
    Output: Ordered list with rank justifications

    Model sees ALL candidates at once -> true apples-to-apples comparison.

    Args:
        viable_jobs: Jobs that passed loose filter
        resume: Full resume dict
        criteria: Search criteria with ranking_priorities
        budget_tracker: Optional budget tracker

    Returns:
        List of jobs with ranking info, sorted by rank
    """
    if not viable_jobs:
        return []

    client = anthropic.Anthropic()

    # Build resume summary
    resume_summary = build_resume_summary_for_ranking(resume)

    # Build ranking prompt
    prompt = build_ranking_prompt(viable_jobs, resume_summary, criteria)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        # Track cost
        if budget_tracker:
            cost = estimate_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
                model="sonnet"
            )
            budget_tracker.add("ranking", cost, {"jobs_ranked": len(viable_jobs)})

        # Extract text
        text_content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text_content += block.text

        # Parse rankings
        ranked_jobs = parse_rankings(text_content, viable_jobs)
        return ranked_jobs

    except Exception as e:
        print(f"    Warning: Ranking failed: {e}")
        # Return jobs in original order with default ranks
        for i, job in enumerate(viable_jobs, 1):
            job['rank'] = i
            job['rank_justification'] = "Ranking failed, using original order"
        return viable_jobs


def build_resume_summary_for_ranking(resume: dict) -> str:
    """Build a comprehensive resume summary for ranking."""
    parts = []

    # Summary
    summary = resume.get('summary', {})
    if isinstance(summary, dict):
        parts.append(f"**Summary**: {summary.get('default', '')}")
    elif isinstance(summary, str):
        parts.append(f"**Summary**: {summary}")

    # Skills
    skills = resume.get('skills', {})
    if isinstance(skills, dict):
        for category, skill_list in skills.items():
            if isinstance(skill_list, list) and skill_list:
                parts.append(f"**{category.title()}**: {', '.join(skill_list)}")
    elif isinstance(skills, list):
        parts.append(f"**Skills**: {', '.join(skills)}")

    # Experience highlights
    experience = resume.get('experience', [])
    if experience:
        parts.append("**Experience**:")
        for exp in experience[:3]:  # Top 3
            company = exp.get('company', '')
            role = exp.get('role', '')
            highlights = exp.get('highlights', [])[:2]  # Top 2 bullets
            parts.append(f"- {role} @ {company}")
            for h in highlights:
                # Handle both string and dict formats
                if isinstance(h, dict):
                    text = h.get('text', '')
                else:
                    text = str(h)
                if text:
                    parts.append(f"  - {text[:100]}...")

    # Education
    education = resume.get('education', [])
    if education:
        edu = education[0]
        parts.append(f"**Education**: {edu.get('degree', '')} from {edu.get('institution', '')}")

    return "\n".join(parts)


def build_ranking_prompt(jobs: list[dict], resume_summary: str, criteria: dict) -> str:
    """Build the ranking prompt."""
    search = criteria.get('search', {})
    priorities = criteria.get('ranking_priorities', {})

    target_roles = search.get('titles', [])
    must_have = search.get('must_have_any', [])

    # Format jobs
    jobs_text = ""
    for i, job in enumerate(jobs, 1):
        jobs_text += f"""
### Job {i}: {job.get('title', 'Unknown')} @ {job.get('company', 'Unknown')}
- ID: {job.get('id', i)}
- Location: {job.get('location', 'Unknown')}
- Filter Score: {job.get('score', 'N/A')}
- Description Preview: {job.get('description_preview', job.get('description', '')[:400])}
"""

    # Format priorities
    priorities_text = ""
    if priorities:
        priorities_text = "## Ranking Criteria (weights)\n"
        for key, weight in priorities.items():
            priorities_text += f"- {key.replace('_', ' ').title()}: {int(weight * 100)}%\n"
    else:
        priorities_text = """## Ranking Criteria (default weights)
- Skills match: 25%
- Role type alignment: 25%
- Relevant focus areas: 20%
- Research vs production balance: 15%
- Company fit: 10%
- Seniority fit: 5%
"""

    prompt = f"""You are ranking job opportunities for a candidate. You will see ALL viable jobs at once and must rank them from most to least relevant.

## Candidate Profile
{resume_summary}

## Candidate Preferences
Target roles: {', '.join(target_roles)}
Key interests: {', '.join(must_have)}

{priorities_text}

## Jobs to Rank
{jobs_text}

## Task
Rank ALL jobs from #1 (best fit) to #{len(jobs)} (worst fit).
Be decisive — avoid ties. Consider the WHOLE pool when ranking.

Return JSON:
{{
  "rankings": [
    {{"rank": 1, "job_id": "...", "company": "...", "role": "...", "justification": "..."}},
    {{"rank": 2, "job_id": "...", "company": "...", "role": "...", "justification": "..."}},
    ...
  ],
  "top_5_summary": "Brief explanation of why the top 5 stand out"
}}

Return ONLY the JSON object."""

    return prompt


def parse_rankings(response_text: str, original_jobs: list[dict]) -> list[dict]:
    """
    Parse ranking response and merge with original job data.

    Args:
        response_text: API response text
        original_jobs: Original job dicts

    Returns:
        Jobs sorted by rank with ranking info added
    """
    # Find JSON in response
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not match:
        # Return original order
        for i, job in enumerate(original_jobs, 1):
            job['rank'] = i
        return original_jobs

    try:
        result = json.loads(match.group())
        rankings = result.get('rankings', [])

        # Build lookup by job_id
        job_lookup = {job.get('id', str(i)): job for i, job in enumerate(original_jobs)}

        # Add ranking info to jobs
        ranked_jobs = []
        for rank_info in rankings:
            job_id = rank_info.get('job_id', '')
            if job_id in job_lookup:
                job = job_lookup[job_id]
                job['rank'] = rank_info.get('rank', 999)
                job['rank_justification'] = rank_info.get('justification', '')
                ranked_jobs.append(job)

        # Add any missing jobs at the end
        ranked_ids = {j.get('id') for j in ranked_jobs}
        for job in original_jobs:
            if job.get('id') not in ranked_ids:
                job['rank'] = len(ranked_jobs) + 1
                job['rank_justification'] = 'Not included in ranking response'
                ranked_jobs.append(job)

        # Sort by rank
        ranked_jobs.sort(key=lambda x: x.get('rank', 999))

        return ranked_jobs

    except json.JSONDecodeError:
        # Return original order
        for i, job in enumerate(original_jobs, 1):
            job['rank'] = i
        return original_jobs
