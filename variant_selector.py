"""
Resume variant selection for jobflow-ai.

Implements a hybrid approach:
1. Rule-based scoring (free) to rank variants by keyword matching
2. AI tiebreaker (Haiku, ~$0.002) when scores are close
3. YAML fallback when no variant matches well

The selected variant provides STRUCTURE (section order, detail level).
AI tailoring still provides WORDING customization (summary, skills order, bullet framing).
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from budget import BudgetTracker


@dataclass
class VariantScore:
    """Score for a single resume variant."""
    name: str
    score: int
    primary_matches: list[str]
    secondary_matches: list[str]
    anti_matches: list[str]
    description: str
    file_path: str
    summary_line: Optional[str]


@dataclass
class ResumeSelection:
    """Result of variant selection."""
    variant: Optional[str]  # None if YAML fallback
    summary_line: Optional[str]  # Pre-written summary line for this variant
    selection_method: str  # "rule_based", "ai_tiebreaker", "ai_thoughtful", or "yaml_fallback"
    scores: list[VariantScore]  # All scores for transparency
    ai_reasoning: Optional[str]  # If AI was used, its reasoning
    ambiguity_level: str = "clear"  # "clear", "moderate", or "high"
    tradeoff_analysis: Optional[str] = None  # For ambiguous cases, explains tradeoffs


def load_variant_meta(meta_path: str = "data/resume_variants/meta.yaml") -> dict:
    """
    Load variant metadata from YAML.

    Args:
        meta_path: Path to the meta.yaml file

    Returns:
        Parsed metadata dict
    """
    meta_file = Path(meta_path)
    if not meta_file.exists():
        raise FileNotFoundError(f"Variant metadata not found: {meta_path}")

    with open(meta_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def score_variant(
    job_description: str,
    variant_name: str,
    variant_config: dict,
    scoring_config: dict
) -> VariantScore:
    """
    Score a single variant against a job description.

    Uses keyword matching to compute a relevance score.

    Args:
        job_description: Full job description text
        variant_name: Name of the variant (e.g., "Applied_AI")
        variant_config: Variant configuration from meta.yaml
        scoring_config: Scoring weights from meta.yaml

    Returns:
        VariantScore with computed score and matched keywords
    """
    text_lower = job_description.lower()

    primary_weight = scoring_config.get('primary_signal_weight', 3)
    secondary_weight = scoring_config.get('secondary_signal_weight', 1)
    anti_weight = scoring_config.get('anti_signal_weight', -5)

    score = 0
    primary_matches = []
    secondary_matches = []
    anti_matches = []

    # Check primary signals
    for signal in variant_config.get('primary_signals', []):
        if signal.lower() in text_lower:
            score += primary_weight
            primary_matches.append(signal)

    # Check secondary signals
    for signal in variant_config.get('secondary_signals', []):
        if signal.lower() in text_lower:
            score += secondary_weight
            secondary_matches.append(signal)

    # Check anti-signals
    for signal in variant_config.get('anti_signals', []):
        if signal.lower() in text_lower:
            score += anti_weight
            anti_matches.append(signal)

    return VariantScore(
        name=variant_name,
        score=score,
        primary_matches=primary_matches,
        secondary_matches=secondary_matches,
        anti_matches=anti_matches,
        description=variant_config.get('description', ''),
        file_path=variant_config.get('file', ''),
        summary_line=variant_config.get('summary_line')
    )


def score_all_variants(
    job_description: str,
    meta_config: dict
) -> list[VariantScore]:
    """
    Score all variants against a job description.

    Args:
        job_description: Full job description text
        meta_config: Full metadata config from meta.yaml

    Returns:
        List of VariantScore sorted by score (highest first)
    """
    variants = meta_config.get('variants', {})
    scoring = meta_config.get('scoring', {})

    scores = []
    for name, config in variants.items():
        score = score_variant(job_description, name, config, scoring)
        scores.append(score)

    # Sort by score descending
    scores.sort(key=lambda x: x.score, reverse=True)
    return scores


def ai_select_variant(
    job_description: str,
    candidates: list[VariantScore],
    budget_tracker: Optional['BudgetTracker'] = None
) -> tuple[str, str]:
    """
    Use Haiku to select between tied candidates.

    Args:
        job_description: Full job description text
        candidates: Top 2-3 candidates with similar scores
        budget_tracker: Optional budget tracker

    Returns:
        Tuple of (selected_variant_name, reasoning)
    """
    # Import here to avoid requiring anthropic for rule-based selection
    import anthropic
    from budget import estimate_cost

    client = anthropic.Anthropic()

    # Build candidate descriptions
    candidate_text = ""
    for c in candidates:
        candidate_text += f"\n{c.name}:\n"
        candidate_text += f"  Description: {c.description}\n"
        candidate_text += f"  Score: {c.score}\n"
        candidate_text += f"  Matched keywords: {', '.join(c.primary_matches + c.secondary_matches)}\n"

    # Truncate job description to save tokens
    jd_truncated = job_description[:3000] if len(job_description) > 3000 else job_description

    prompt = f"""Select the best resume variant for this job posting.

## Job Description (truncated)
{jd_truncated}

## Resume Variants to Choose From
{candidate_text}

## Task
Choose the SINGLE best variant for this job. Consider:
1. Which variant's focus best matches the role's primary requirements?
2. Which variant's structure (section order, emphasis) fits the role type?
3. Are there any red flags that make a variant unsuitable?

Respond with ONLY a JSON object:
{{"selected": "variant_name", "reasoning": "one sentence explanation"}}
"""

    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )

    # Track cost
    if budget_tracker:
        cost = estimate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            model="haiku"
        )
        budget_tracker.add("variant_selection", cost, {
            "candidates": [c.name for c in candidates]
        })

    # Parse response
    text = response.content[0].text

    # Extract JSON from response
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        import json
        result = json.loads(match.group())
        return result.get('selected', candidates[0].name), result.get('reasoning', '')

    # Fallback to first candidate if parsing fails
    return candidates[0].name, "Fallback: Could not parse AI response"


def ai_select_variant_thoughtful(
    job_description: str,
    candidates: list[VariantScore],
    budget_tracker: Optional['BudgetTracker'] = None
) -> tuple[str, str, str]:
    """
    For highly ambiguous cases, provide deeper analysis with tradeoffs.

    Used when variant scores are very close (gap < 5), indicating genuine
    ambiguity about which variant is best.

    Args:
        job_description: Full job description text
        candidates: Top 2-3 candidates with very similar scores
        budget_tracker: Optional budget tracker

    Returns:
        Tuple of (selected_variant_name, reasoning, tradeoff_analysis)
    """
    import json
    import anthropic
    from budget import estimate_cost

    client = anthropic.Anthropic()

    # Build candidate descriptions
    candidate_text = ""
    for c in candidates:
        candidate_text += f"\n**{c.name}** (score: {c.score}):\n"
        candidate_text += f"  Description: {c.description}\n"
        candidate_text += f"  Primary matches: {', '.join(c.primary_matches) if c.primary_matches else 'none'}\n"
        candidate_text += f"  Secondary matches: {', '.join(c.secondary_matches) if c.secondary_matches else 'none'}\n"
        candidate_text += f"  Anti-signals: {', '.join(c.anti_matches) if c.anti_matches else 'none'}\n"

    # Truncate job description to save tokens
    jd_truncated = job_description[:3000] if len(job_description) > 3000 else job_description

    prompt = f"""This is a GENUINELY AMBIGUOUS case. Multiple resume variants scored similarly.

## Job Description
{jd_truncated}

## Close Candidates
{candidate_text}

## Task
1. Acknowledge this is a difficult choice - the scores are very close
2. Analyze trade-offs for EACH candidate variant
3. Make a selection with explicit confidence level
4. Explain what information would help make a more confident choice

Return a JSON object:
{{
  "selected": "variant_name",
  "confidence": "low|medium|high",
  "reasoning": "2-3 sentences explaining the selection",
  "tradeoffs": {{
    "variant1_name": "Pro: X. Con: Y.",
    "variant2_name": "Pro: A. Con: B."
  }},
  "what_would_help": "What info would make this clearer (e.g., team composition, research vs production focus)"
}}
"""

    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    # Track cost
    if budget_tracker:
        cost = estimate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            model="haiku"
        )
        budget_tracker.add("variant_selection_thoughtful", cost, {
            "candidates": [c.name for c in candidates],
            "ambiguity": "high"
        })

    # Parse response
    text = response.content[0].text

    # Extract JSON from response
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            selected = result.get('selected', candidates[0].name)
            confidence = result.get('confidence', 'medium')
            reasoning = result.get('reasoning', '')
            tradeoffs = result.get('tradeoffs', {})
            what_would_help = result.get('what_would_help', '')

            # Format tradeoff analysis
            tradeoff_lines = []
            for variant, analysis in tradeoffs.items():
                tradeoff_lines.append(f"- {variant}: {analysis}")
            if what_would_help:
                tradeoff_lines.append(f"\nMore info needed: {what_would_help}")

            tradeoff_analysis = '\n'.join(tradeoff_lines)
            full_reasoning = f"[{confidence} confidence] {reasoning}"

            return selected, full_reasoning, tradeoff_analysis
        except json.JSONDecodeError:
            pass

    # Fallback to first candidate if parsing fails
    return candidates[0].name, "Fallback: Could not parse AI response", ""


def select_resume_variant(
    job_description: str,
    meta_path: str = "data/resume_variants/meta.yaml",
    budget_tracker: Optional['BudgetTracker'] = None
) -> ResumeSelection:
    """
    Select the best resume variant for a job description.

    Three-stage process:
    1. Score all variants by keyword matching (free)
    2. If clear winner, auto-select; if tie, use Haiku (~$0.002)
    3. If no good match, return None for YAML fallback

    Args:
        job_description: Full job description text
        meta_path: Path to variant metadata
        budget_tracker: Optional budget tracker

    Returns:
        ResumeSelection with variant info or None for YAML fallback
    """
    # Load metadata
    meta_config = load_variant_meta(meta_path)
    thresholds = meta_config.get('thresholds', {})

    min_score = thresholds.get('min_candidate_score', 5)
    auto_select_gap = thresholds.get('auto_select_threshold', 10)

    # Score all variants
    scores = score_all_variants(job_description, meta_config)

    if not scores:
        return ResumeSelection(
            variant=None,
            summary_line=None,
            selection_method="yaml_fallback",
            scores=[],
            ai_reasoning="No variants configured"
        )

    top = scores[0]
    second = scores[1] if len(scores) > 1 else None

    # Check if top score meets minimum threshold
    if top.score < min_score:
        return ResumeSelection(
            variant=None,
            summary_line=None,
            selection_method="yaml_fallback",
            scores=scores,
            ai_reasoning=f"Top score ({top.score}) below threshold ({min_score})",
            ambiguity_level="clear",
            tradeoff_analysis=None
        )

    # Calculate gap to determine ambiguity level
    gap = top.score - (second.score if second else 0)

    # Determine selection method and ambiguity level
    selected_name = None
    ai_reasoning = None
    selection_method = "rule_based"
    ambiguity_level = "clear"
    tradeoff_analysis = None

    if second is None or gap >= auto_select_gap:
        # Clear winner - auto select
        selected_name = top.name
        selection_method = "rule_based"
        ambiguity_level = "clear"
    elif gap >= 5:
        # Moderate ambiguity - standard AI tiebreaker
        close_candidates = [s for s in scores if s.score >= top.score - auto_select_gap][:3]

        if len(close_candidates) > 1:
            selected_name, ai_reasoning = ai_select_variant(
                job_description,
                close_candidates,
                budget_tracker
            )
            selection_method = "ai_tiebreaker"
            ambiguity_level = "moderate"
        else:
            selected_name = top.name
            selection_method = "rule_based"
    else:
        # High ambiguity (gap < 5) - thoughtful selection with tradeoff analysis
        close_candidates = [s for s in scores if s.score >= top.score - auto_select_gap][:3]

        if len(close_candidates) > 1:
            selected_name, ai_reasoning, tradeoff_analysis = ai_select_variant_thoughtful(
                job_description,
                close_candidates,
                budget_tracker
            )
            selection_method = "ai_thoughtful"
            ambiguity_level = "high"
        else:
            selected_name = top.name
            selection_method = "rule_based"
            ambiguity_level = "clear"

    # Get the selected variant's summary line
    selected_score = next((s for s in scores if s.name == selected_name), top)

    return ResumeSelection(
        variant=selected_name,
        summary_line=selected_score.summary_line,
        selection_method=selection_method,
        scores=scores,
        ai_reasoning=ai_reasoning,
        ambiguity_level=ambiguity_level,
        tradeoff_analysis=tradeoff_analysis
    )


def format_selection_summary(selection: ResumeSelection) -> str:
    """
    Format a human-readable summary of the selection.

    Args:
        selection: ResumeSelection result

    Returns:
        Formatted string for logging/display
    """
    lines = ["Resume Variant Selection:"]
    lines.append(f"  Method: {selection.selection_method}")
    lines.append(f"  Ambiguity: {selection.ambiguity_level}")

    if selection.variant:
        lines.append(f"  Selected: {selection.variant}")
        if selection.ai_reasoning:
            lines.append(f"  AI Reasoning: {selection.ai_reasoning}")
    else:
        lines.append("  Selected: YAML fallback (no variant matched)")
        if selection.ai_reasoning:
            lines.append(f"  Reason: {selection.ai_reasoning}")

    # Show tradeoff analysis for ambiguous cases
    if selection.tradeoff_analysis:
        lines.append("\n  Tradeoff Analysis:")
        for line in selection.tradeoff_analysis.split('\n'):
            lines.append(f"    {line}")

    lines.append("\n  Score Ranking:")
    for score in selection.scores[:5]:  # Show top 5
        lines.append(f"    {score.name}: {score.score}")
        if score.primary_matches:
            lines.append(f"      Primary: {', '.join(score.primary_matches[:3])}")
        if score.anti_matches:
            lines.append(f"      Anti: {', '.join(score.anti_matches)}")

    return '\n'.join(lines)
