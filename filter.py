"""
Job filtering module for jobflow-ai.

Two-stage filtering:
1. Rule-based pre-filter (free, deterministic)
2. Loose AI filter using Haiku (cheap, removes borderline cases)
"""

import json
import re
from typing import Optional

import anthropic

from budget import BudgetTracker, estimate_cost


def rule_filter(job: dict, criteria: dict) -> tuple[bool, Optional[str]]:
    """
    Deterministic elimination using string matching.

    Args:
        job: Job dict with title, description, location
        criteria: Search criteria dict

    Returns:
        Tuple of (passed: bool, rejection_reason: str | None)
    """
    title = job.get('title', '').lower()
    description = job.get('description', job.get('description_preview', '')).lower()
    text = f"{title} {description}"

    search_criteria = criteria.get('search', {})
    rule_config = criteria.get('rule_filter', {})

    # Check exclude keywords
    exclude_if = search_criteria.get('exclude_if', [])
    for keyword in exclude_if:
        if keyword.lower() in text:
            return False, f"Contains excluded keyword: {keyword}"

    # Check title match (at least one target title should be present)
    target_titles = search_criteria.get('titles', [])
    if target_titles:
        title_matches = any(t.lower() in title for t in target_titles)
        if not title_matches:
            # Check for partial matches (e.g., "Engineer" in "Software Engineer")
            title_words = [t.lower().split()[-1] for t in target_titles]  # Last word
            partial_match = any(w in title for w in title_words)
            if not partial_match:
                return False, f"Title '{job.get('title', 'Unknown')}' doesn't match target roles"

    # Check location (if strict mode)
    if rule_config.get('location_strict', False):
        locations = search_criteria.get('locations', [])
        job_location = job.get('location', '').lower()
        location_ok = any(loc.lower() in job_location or loc.lower() in text for loc in locations)
        if not location_ok and 'remote' not in job_location:
            return False, f"Location '{job.get('location', 'Unknown')}' doesn't match target locations"

    # Check seniority excludes
    seniority_exclude = rule_config.get('seniority_exclude', [])
    for level in seniority_exclude:
        if level.lower() in title:
            return False, f"Seniority mismatch: {level}"

    return True, None


def apply_rule_filter(jobs: list[dict], criteria: dict) -> tuple[list[dict], list[dict]]:
    """
    Apply rule filter to all jobs.

    Args:
        jobs: List of job dicts
        criteria: Search criteria dict

    Returns:
        Tuple of (passed_jobs, rejection_log)
    """
    passed = []
    rejected = []

    for job in jobs:
        ok, reason = rule_filter(job, criteria)
        if ok:
            passed.append(job)
        else:
            rejected.append({
                "job_id": job.get('id', ''),
                "title": job.get('title', ''),
                "company": job.get('company', ''),
                "reason": reason
            })

    return passed, rejected


def filter_jobs_loose(
    jobs: list[dict],
    resume: dict,
    criteria: dict,
    budget_tracker: Optional[BudgetTracker] = None,
    batch_size: int = 5
) -> list[dict]:
    """
    Loose relevance filter using Claude Haiku (batched).

    Purpose: Eliminate borderline cases that need AI judgment.
    NOT for final ranking — scores may be inconsistent across batches.

    Args:
        jobs: Jobs that passed rule_filter
        resume: Resume dict (just need summary for filtering)
        criteria: Search criteria
        budget_tracker: Optional budget tracker
        batch_size: Jobs per API call

    Returns:
        Jobs with scores added
    """
    if not jobs:
        return []

    client = anthropic.Anthropic()

    # Build resume summary (brief, for filtering)
    resume_summary = build_resume_summary(resume)

    # Process in batches
    scored_jobs = []
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        batch_scores = score_job_batch(client, batch, resume_summary, criteria, budget_tracker)

        for job, score_info in zip(batch, batch_scores):
            job['score'] = score_info.get('score', 0)
            job['score_reason'] = score_info.get('reason', '')
            scored_jobs.append(job)

    return scored_jobs


def build_resume_summary(resume: dict) -> str:
    """Build a brief resume summary for filtering."""
    summary_parts = []

    # Contact
    contact = resume.get('contact', {})
    if contact.get('name'):
        summary_parts.append(f"Name: {contact['name']}")

    # Summary
    summary = resume.get('summary', {})
    if isinstance(summary, dict):
        summary_parts.append(f"Summary: {summary.get('default', '')[:200]}")
    elif isinstance(summary, str):
        summary_parts.append(f"Summary: {summary[:200]}")

    # Skills
    skills = resume.get('skills', {})
    if isinstance(skills, dict):
        all_skills = []
        for category, skill_list in skills.items():
            if isinstance(skill_list, list):
                all_skills.extend(skill_list)
        summary_parts.append(f"Skills: {', '.join(all_skills[:15])}")
    elif isinstance(skills, list):
        summary_parts.append(f"Skills: {', '.join(skills[:15])}")

    # Experience level
    experience = resume.get('experience', [])
    if experience:
        years = len(experience)
        companies = [e.get('company', '') for e in experience[:3]]
        summary_parts.append(f"Experience: ~{years} positions at {', '.join(companies)}")

    return "\n".join(summary_parts)


def score_job_batch(
    client: anthropic.Anthropic,
    jobs: list[dict],
    resume_summary: str,
    criteria: dict,
    budget_tracker: Optional[BudgetTracker] = None
) -> list[dict]:
    """
    Score a batch of jobs using Haiku.

    Args:
        client: Anthropic client
        jobs: Batch of jobs to score
        resume_summary: Brief resume summary
        criteria: Search criteria
        budget_tracker: Optional budget tracker

    Returns:
        List of score dicts with score and reason
    """
    search = criteria.get('search', {})
    target_roles = search.get('titles', [])
    must_have = search.get('must_have_any', [])

    # Format jobs for prompt
    jobs_text = ""
    for i, job in enumerate(jobs, 1):
        jobs_text += f"""
Job {i}:
- ID: {job.get('id', i)}
- Title: {job.get('title', 'Unknown')}
- Company: {job.get('company', 'Unknown')}
- Location: {job.get('location', 'Unknown')}
- Preview: {job.get('description_preview', job.get('description', '')[:300])}
"""

    prompt = f"""Quickly score these job postings for basic fit. Be generous — we'll rank properly later.

## Candidate Summary
{resume_summary}

## Criteria
Target roles: {', '.join(target_roles)}
Bonus if contains: {', '.join(must_have)}

## Jobs to Score
{jobs_text}

Score each job 1-10. Keep threshold LOW (5+) — we just want to remove obvious mismatches.

Return a JSON array with one entry per job:
[
  {{"job_id": "...", "score": 6, "reason": "Brief reason"}},
  ...
]

Return ONLY the JSON array."""

    try:
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract text
        text_content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text_content += block.text

        # Track cost
        if budget_tracker:
            cost = estimate_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
                model="haiku"
            )
            budget_tracker.add("filter_batch", cost, {"batch_size": len(jobs)})

        # Parse response
        match = re.search(r'\[.*\]', text_content, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            # Ensure we have scores for all jobs
            if len(scores) == len(jobs):
                return scores

        # Fallback: return default scores
        return [{"score": 5, "reason": "Default score"} for _ in jobs]

    except Exception as e:
        print(f"    Warning: Scoring batch failed: {e}")
        return [{"score": 5, "reason": "Scoring failed"} for _ in jobs]
