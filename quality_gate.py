"""
Quality gate module for jobflow-ai.

Implements an LLM-as-Judge pattern to review generated resumes
from a recruiter's perspective and provide improvement suggestions.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResumeReview:
    """Result of resume quality review."""
    overall_score: int  # 0-100
    relevance: int  # 0-10: Does it target this specific role?
    impact: int  # 0-10: Do bullets show measurable results?
    clarity: int  # 0-10: Is it well-structured and scannable?
    specificity: int  # 0-10: Does it avoid generic language?
    improvements: list[str]  # Specific improvement suggestions
    passes_threshold: bool  # True if overall_score >= threshold
    raw_response: dict  # Full response for debugging


def load_review_prompt() -> str:
    """Load the review prompt template."""
    prompt_path = Path(__file__).parent / "prompts" / "review.md"
    return prompt_path.read_text(encoding='utf-8')


def review_resume_as_recruiter(
    resume_text: str,
    job_description: str,
    company: str,
    role: str,
    client,  # anthropic.Anthropic
    budget_tracker=None,
    threshold: int = 60
) -> ResumeReview:
    """
    Have LLM review resume from a recruiter's perspective.

    Uses Sonnet to evaluate the resume against the job description
    and provide structured feedback with improvement suggestions.

    Args:
        resume_text: Full text content of the resume
        job_description: Job description to evaluate against
        company: Company name
        role: Role title
        client: Anthropic client
        budget_tracker: Optional budget tracker
        threshold: Minimum score to pass (default 60)

    Returns:
        ResumeReview with scores and improvement suggestions
    """
    from budget import estimate_cost

    # Load and format prompt
    template = load_review_prompt()
    prompt = template.replace("{company}", company)
    prompt = prompt.replace("{role}", role)
    prompt = prompt.replace("{job_description}", job_description[:4000])  # Truncate
    prompt = prompt.replace("{resume_text}", resume_text)

    # Call LLM
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    # Track cost
    if budget_tracker:
        cost = estimate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            model="sonnet"
        )
        budget_tracker.add("quality_review", cost, {
            "company": company,
            "role": role
        })

    # Parse response
    text = response.content[0].text

    # Extract JSON from response
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        logger.warning("Could not parse quality review response")
        return ResumeReview(
            overall_score=50,
            relevance=5,
            impact=5,
            clarity=5,
            specificity=5,
            improvements=["Could not parse review response"],
            passes_threshold=False,
            raw_response={"error": "parse_failed", "text": text}
        )

    try:
        result = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error in quality review: {e}")
        return ResumeReview(
            overall_score=50,
            relevance=5,
            impact=5,
            clarity=5,
            specificity=5,
            improvements=["Could not parse review response"],
            passes_threshold=False,
            raw_response={"error": str(e), "text": text}
        )

    # Extract scores
    relevance = result.get('relevance', 5)
    impact = result.get('impact', 5)
    clarity = result.get('clarity', 5)
    specificity = result.get('specificity', 5)
    improvements = result.get('improvements', [])

    # Calculate overall score (0-100)
    overall_score = int((relevance + impact + clarity + specificity) * 2.5)

    return ResumeReview(
        overall_score=overall_score,
        relevance=relevance,
        impact=impact,
        clarity=clarity,
        specificity=specificity,
        improvements=improvements,
        passes_threshold=overall_score >= threshold,
        raw_response=result
    )


def format_review_summary(review: ResumeReview) -> str:
    """
    Format a human-readable summary of the review.

    Args:
        review: ResumeReview result

    Returns:
        Formatted string for logging/display
    """
    status = "PASS" if review.passes_threshold else "NEEDS IMPROVEMENT"

    lines = [
        f"Resume Quality Review: {status}",
        f"  Overall Score: {review.overall_score}/100",
        f"  - Relevance: {review.relevance}/10",
        f"  - Impact: {review.impact}/10",
        f"  - Clarity: {review.clarity}/10",
        f"  - Specificity: {review.specificity}/10",
    ]

    if review.improvements:
        lines.append("\n  Improvements:")
        for imp in review.improvements[:5]:  # Show top 5
            lines.append(f"    - {imp}")

    return '\n'.join(lines)
