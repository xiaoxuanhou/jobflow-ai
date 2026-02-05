"""
Processing module for jobflow-ai.

Handles job analysis, resume tailoring, and outreach generation as separate steps.
Uses Claude API with web search for LinkedIn profile discovery.
"""

import json
import re
from pathlib import Path
from typing import Optional

import anthropic
import yaml

from budget import BudgetTracker, estimate_cost, COSTS
from utils import (
    render_docx,
    convert_to_pdf,
    extract_text_from_docx,
    render_outreach_md,
    save_model_inputs,
    save_json,
    get_output_dir,
    slugify,
)
from variant_selector import (
    select_resume_variant,
    format_selection_summary,
    ResumeSelection,
)
from quality_gate import (
    review_resume_as_recruiter,
    format_review_summary,
)


# =============================================================================
# Prompt Loading
# =============================================================================

def load_prompt_template(name: str = "process") -> str:
    """Load a prompt template by name."""
    prompt_path = Path(__file__).parent / "prompts" / f"{name}.md"
    return prompt_path.read_text(encoding='utf-8')


# =============================================================================
# Resume Tailoring
# =============================================================================

def build_tailoring_prompt(
    job_description: str,
    resume: dict,
    variant_selection: Optional[ResumeSelection] = None
) -> str:
    """
    Build the resume tailoring prompt.

    Args:
        job_description: Full job description text
        resume: Resume dict (from YAML)
        variant_selection: Optional variant selection result

    Returns:
        Complete prompt string
    """
    template = load_prompt_template("process")

    # Format resume as YAML for readability
    resume_yaml = yaml.dump(resume, default_flow_style=False, allow_unicode=True)

    # Format variant context
    if variant_selection and variant_selection.variant:
        variant_context = f"""A resume variant has been PRE-SELECTED for this job:
- Variant: {variant_selection.variant}
- Selection method: {variant_selection.selection_method}
- Ambiguity level: {variant_selection.ambiguity_level}
- Existing summary line: {variant_selection.summary_line or 'None'}

This variant provides the BASE STRUCTURE. Your task is MINIMAL, TARGETED modifications:
1. Summary: Provide a new summary ONLY if the existing one needs significant adjustment
2. Skills order: Provide priority order ONLY if reordering would improve relevance
3. skills_to_add: List skills from YAML that are missing from the variant but relevant to this job
4. DO NOT rewrite experience bullets - the variant's bullets are well-crafted"""

        # Add tradeoff analysis for ambiguous cases
        if variant_selection.ambiguity_level == "high" and variant_selection.tradeoff_analysis:
            variant_context += f"""

## Variant Selection Ambiguity Note
This variant was selected from close alternatives. Consider these tradeoffs in your tailoring:
{variant_selection.tradeoff_analysis}

Where possible, incorporate strengths from alternative approaches while maintaining the selected variant's structure."""
    else:
        variant_context = """No variant selected - using YAML fallback.
Generate full tailored content including reframed experience bullets."""

    # Replace placeholders
    prompt = template.replace("{job_description}", job_description)
    prompt = prompt.replace("{resume_yaml}", resume_yaml)
    prompt = prompt.replace("{variant_context}", variant_context)

    return prompt


def call_claude(
    client: anthropic.Anthropic,
    prompt: str,
    budget_tracker: Optional[BudgetTracker] = None,
    stage_name: str = "api_call"
) -> tuple[dict, int, int]:
    """
    Call Claude API and parse structured JSON response.

    Args:
        client: Anthropic client
        prompt: Full prompt string
        budget_tracker: Optional budget tracker
        stage_name: Name for budget tracking

    Returns:
        Tuple of (parsed_response, input_tokens, output_tokens)
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    # Extract text content
    text_content = ""
    for block in response.content:
        if hasattr(block, 'text'):
            text_content += block.text

    # Parse JSON from response
    match = re.search(r'\{.*\}', text_content, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response")

    json_str = match.group()

    # Fix LaTeX backslashes that aren't properly escaped for JSON
    def fix_latex_escapes(s):
        valid_escapes = set('nrtbf"\\/u')
        result = []
        i = 0
        while i < len(s):
            if s[i] == '\\' and i + 1 < len(s):
                next_char = s[i + 1]
                if next_char not in valid_escapes:
                    result.append('\\\\')
                    i += 1
                else:
                    result.append(s[i])
                    i += 1
            else:
                result.append(s[i])
                i += 1
        return ''.join(result)

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        fixed_json = fix_latex_escapes(json_str)
        result = json.loads(fixed_json)

    # Track cost
    if budget_tracker:
        cost = estimate_cost(input_tokens, output_tokens, model="sonnet")
        budget_tracker.add(stage_name, cost, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        })

    return result, input_tokens, output_tokens


# =============================================================================
# Outreach Generation
# =============================================================================

def build_outreach_prompt(
    company: str,
    role: str,
    candidate_summary: str,
    candidate_strengths: list[str],
    profiles: list[dict],
    outreach_context: dict
) -> str:
    """
    Build the outreach generation prompt.

    Args:
        company: Company name
        role: Role title
        candidate_summary: Brief candidate summary
        candidate_strengths: List of strengths relevant to this role
        profiles: List of LinkedIn profiles found
        outreach_context: Dict with schools, previous_companies, etc.

    Returns:
        Complete prompt string
    """
    template = load_prompt_template("outreach")

    # Format profiles
    if profiles:
        profiles_text = json.dumps(profiles, indent=2)
    else:
        profiles_text = "No LinkedIn profiles found."

    # Format outreach context
    outreach_context_text = yaml.dump(outreach_context, default_flow_style=False)

    # Format strengths
    strengths_text = "\n".join(f"- {s}" for s in candidate_strengths) if candidate_strengths else "See resume"

    # Replace placeholders
    prompt = template.replace("{company}", company)
    prompt = prompt.replace("{role}", role)
    prompt = prompt.replace("{candidate_summary}", candidate_summary)
    prompt = prompt.replace("{candidate_strengths}", strengths_text)
    prompt = prompt.replace("{profiles}", profiles_text)
    prompt = prompt.replace("{outreach_context}", outreach_context_text)

    return prompt


def search_linkedin_profiles(
    client: anthropic.Anthropic,
    company: str,
    outreach_context: dict,
    budget_tracker: Optional[BudgetTracker] = None
) -> list[dict]:
    """
    Search for LinkedIn profiles at the target company.

    Uses web search to find relevant contacts based on:
    - Alumni from same schools
    - People from same previous companies
    - Team members and recruiters

    Args:
        client: Anthropic client
        company: Target company name
        outreach_context: Dict with schools, previous_companies, etc.
        budget_tracker: Optional budget tracker

    Returns:
        List of profile dicts with name, title, url, connection_type
    """
    schools = outreach_context.get('schools', [])
    previous_companies = outreach_context.get('previous_companies', [])

    profiles = []
    search_queries = []

    # Build search queries
    for school in schools[:2]:
        search_queries.append(f'site:linkedin.com/in "{company}" "{school}"')

    for prev_company in previous_companies[:2]:
        search_queries.append(f'site:linkedin.com/in "{company}" "{prev_company}"')

    search_queries.append(f'site:linkedin.com/in "{company}" recruiter')

    # Search for profiles
    for query in search_queries:
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{
                    "role": "user",
                    "content": f"""Search LinkedIn for people at {company}. Query: {query}

Return a JSON array of profiles found. For each profile include:
- name: Full name
- title: Current job title
- linkedin_url: LinkedIn profile URL
- connection_type: "alumni", "previous_coworker", "recruiter", or "team_member"

Return ONLY the JSON array, nothing else. If no profiles found, return []."""
                }]
            )

            # Extract text content
            text_content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    text_content += block.text

            # Parse JSON
            if text_content.strip():
                match = re.search(r'\[.*\]', text_content, re.DOTALL)
                if match:
                    found_profiles = json.loads(match.group())
                    profiles.extend(found_profiles)

            # Track cost
            if budget_tracker:
                cost = estimate_cost(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    model="sonnet"
                )
                budget_tracker.add("profile_search", cost, {"query": query})

        except Exception as e:
            print(f"Warning: Profile search failed for query '{query}': {e}")
            continue

    # Deduplicate by URL
    seen_urls = set()
    unique_profiles = []
    for profile in profiles:
        url = profile.get('linkedin_url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_profiles.append(profile)

    return unique_profiles[:10]


# =============================================================================
# LaTeX Generation
# =============================================================================

def build_render_data(resume: dict, tailored_content: dict) -> dict:
    """
    Build the render data dictionary for the LaTeX template.

    Merges base resume data with AI-tailored content.

    Args:
        resume: Full resume dict from YAML
        tailored_content: AI-generated tailored content

    Returns:
        Dict ready for Jinja2 template rendering
    """
    contact = resume.get('contact', {})
    summary = tailored_content.get('summary') or resume.get('summary', {}).get('default', '')

    # Experiences
    tailored_experiences = tailored_content.get('experiences', [])
    base_experiences = resume.get('experience', [])

    experiences = []
    if tailored_experiences:
        for exp in tailored_experiences:
            experiences.append({
                'company': exp.get('company', ''),
                'role': exp.get('role', ''),
                'location': exp.get('location', ''),
                'start_date': exp.get('start_date', ''),
                'end_date': exp.get('end_date', ''),
                'bullets': exp.get('bullets', []),
            })
    else:
        for exp in base_experiences:
            bullets = []
            for h in exp.get('highlights', []):
                if isinstance(h, dict):
                    bullets.append(h.get('text', ''))
                else:
                    bullets.append(str(h))
            experiences.append({
                'company': exp.get('company', ''),
                'role': exp.get('role', ''),
                'location': exp.get('location', ''),
                'start_date': exp.get('start_date', ''),
                'end_date': exp.get('end_date', ''),
                'bullets': bullets,
            })

    # Education
    education = []
    for edu in resume.get('education', []):
        education.append({
            'institution': edu.get('institution', ''),
            'degree': edu.get('degree', ''),
            'graduation': edu.get('graduation', ''),
            'start_year': edu.get('start_year', ''),
            'location': edu.get('location', ''),
            'focus': edu.get('focus', ''),
            'advisors': edu.get('advisors', []),
        })

    # Skills
    skills_order = tailored_content.get('skills_order', [])
    base_skills = resume.get('skills', {})

    skills = []
    if skills_order:
        for category_key in skills_order:
            if category_key in base_skills:
                skill_items = base_skills[category_key]
                name = category_key.replace('_', ' ').replace(' and ', ' & ').title()
                skills.append({
                    'name': name,
                    'skill_list': skill_items if isinstance(skill_items, list) else [skill_items],
                })
    else:
        for category, skill_items in base_skills.items():
            name = category.replace('_', ' ').replace(' and ', ' & ').title()
            skills.append({
                'name': name,
                'skill_list': skill_items if isinstance(skill_items, list) else [skill_items],
            })

    # Publications
    publications = []
    pubs = resume.get('publications', {})

    for pub in pubs.get('conference_papers', []):
        publications.append({
            'first_author': 'co-first' in str(pub.get('authors', []))[:50].lower() or pub.get('authors', [''])[0].startswith('Hou'),
            'status': pub.get('status', ''),
            'venue': pub.get('venue', ''),
            'short_title': pub.get('title', '').split(':')[0] if ':' in pub.get('title', '') else pub.get('title', '')[:60],
            'url': pub.get('arxiv', pub.get('ssrn', '')),
            'url_label': 'arXiv' if pub.get('arxiv') else 'SSRN' if pub.get('ssrn') else 'Link',
        })

    for pub in pubs.get('journal_papers', [])[:2]:
        publications.append({
            'first_author': pub.get('authors', [''])[0].startswith('Hou'),
            'status': pub.get('status', ''),
            'venue': pub.get('venue', ''),
            'short_title': pub.get('title', '').split(':')[0] if ':' in pub.get('title', '') else pub.get('title', '')[:60],
            'url': pub.get('ssrn', pub.get('arxiv', '')),
            'url_label': 'SSRN' if pub.get('ssrn') else 'arXiv' if pub.get('arxiv') else 'Link',
        })

    for pub in pubs.get('work_in_progress', [])[:1]:
        publications.append({
            'first_author': pub.get('authors', [''])[0].startswith('Hou'),
            'status': 'Work in progress',
            'venue': '',
            'short_title': pub.get('title', '')[:60],
            'url': '',
            'url_label': '',
        })

    return {
        'contact': contact,
        'summary': summary,
        'experiences': experiences,
        'education': education,
        'skills': skills,
        'publications': publications,
    }


def generate_docx_resume(
    resume: dict,
    tailored_content: dict,
    output_path: str
) -> str:
    """
    Generate a DOCX resume from YAML data and tailored content.

    Args:
        resume: Resume dict (from YAML)
        tailored_content: AI-generated tailoring suggestions
        output_path: Path to save the .docx file

    Returns:
        Path to generated .docx file
    """
    render_data = build_render_data(resume, tailored_content)
    return render_docx(render_data, output_path)


# =============================================================================
# Main Processing Function
# =============================================================================

def process_job(
    job_description: str,
    job_url: str,
    resume: dict,
    budget_tracker: Optional[BudgetTracker] = None,
    output_dir: str = "outputs",
    skip_profiles: bool = False,
    skip_outreach: bool = False,
    variant_meta_path: str = "data/resume_variants/meta.yaml",
    company: Optional[str] = None,
    role_title: Optional[str] = None
) -> dict:
    """
    Process a single job: analyze, tailor resume, generate outreach.

    Args:
        job_description: Full job description text
        job_url: URL of the job posting
        resume: Resume dict (from YAML)
        budget_tracker: Optional budget tracker
        output_dir: Base output directory
        skip_profiles: Skip LinkedIn profile search
        skip_outreach: Skip outreach generation entirely
        variant_meta_path: Path to variant metadata YAML
        company: Optional company name (if known from discovery)
        role_title: Optional role title (if known from discovery)

    Returns:
        Dict with all processing results
    """
    client = anthropic.Anthropic()

    # Use provided company or extract from job description
    if not company or company == "Unknown":
        company = extract_company_name(job_description)
    total_cost = 0.0

    # =========================================================================
    # Step 1: Select resume variant
    # =========================================================================
    print(f"  Selecting resume variant...")
    variant_selection = None
    try:
        variant_selection = select_resume_variant(
            job_description,
            meta_path=variant_meta_path,
            budget_tracker=budget_tracker
        )
        print(format_selection_summary(variant_selection))
    except FileNotFoundError:
        print(f"  Warning: Variant metadata not found, using YAML fallback")
    except Exception as e:
        print(f"  Warning: Variant selection failed: {e}, using YAML fallback")

    # =========================================================================
    # Step 2: Resume tailoring (API call 1)
    # =========================================================================
    print(f"  Analyzing job and tailoring resume...")
    tailoring_prompt = build_tailoring_prompt(job_description, resume, variant_selection)

    tailoring_result, tailor_input, tailor_output = call_claude(
        client, tailoring_prompt, budget_tracker, "tailoring"
    )

    # Extract components (new flattened structure)
    analysis = tailoring_result.get('analysis', {})
    strategy = tailoring_result.get('strategy', {})
    tailored_content = tailoring_result.get('tailored_content', {})

    # Get role - prefer provided title, fall back to analysis
    role = role_title if role_title else analysis.get('role', 'Unknown Role')
    if not analysis.get('company') or analysis.get('company') == 'Unknown':
        analysis['company'] = company
    if not analysis.get('role') or analysis.get('role') == 'Unknown':
        analysis['role'] = role

    tailor_cost = estimate_cost(tailor_input, tailor_output, model="sonnet")
    total_cost += tailor_cost

    # =========================================================================
    # Step 3: Create output directory
    # =========================================================================
    job_output_dir = get_output_dir(company, role, output_dir)
    job_slug = slugify(f"{company}-{role}")

    # =========================================================================
    # Step 4: Generate DOCX resume
    # =========================================================================
    docx_path = job_output_dir / "resume.docx"

    try:
        if variant_selection and variant_selection.variant:
            print(f"  Using variant context: {variant_selection.variant}")
        else:
            print(f"  Using YAML fallback")

        generate_docx_resume(resume, tailored_content, str(docx_path))
        print(f"  Generated: {docx_path}")

        pdf_path = convert_to_pdf(str(docx_path))
        if pdf_path:
            print(f"  Generated: {pdf_path}")
    except Exception as e:
        print(f"  Warning: DOCX generation failed: {e}")

    # =========================================================================
    # Step 5: Quality Gate - Review resume as recruiter
    # =========================================================================
    quality_review = None
    regenerated = False

    if docx_path.exists():
        print(f"  Running quality gate...")
        try:
            resume_text = extract_text_from_docx(str(docx_path))
            quality_review = review_resume_as_recruiter(
                resume_text=resume_text,
                job_description=job_description,
                company=company,
                role=role,
                client=client,
                budget_tracker=budget_tracker,
                threshold=60
            )

            review_cost = estimate_cost(
                quality_review.raw_response.get('input_tokens', 1000),
                quality_review.raw_response.get('output_tokens', 500),
                model="sonnet"
            ) if 'input_tokens' in quality_review.raw_response else 0.01
            total_cost += review_cost

            print(format_review_summary(quality_review))

            # Auto-regenerate if score < 60 (max 1 retry)
            if not quality_review.passes_threshold and quality_review.improvements:
                print(f"  Score below threshold. Regenerating with improvements...")
                regenerated = True

                # Add improvements to tailoring context
                improvement_context = "\n".join(
                    f"- {imp}" for imp in quality_review.improvements[:5]
                )

                # Re-run tailoring with improvement context
                enhanced_prompt = tailoring_prompt + f"""

## Quality Improvement Required
A recruiter review found the following issues. Address these in your tailored content:
{improvement_context}

Focus on making bullets more impactful with metrics and specific outcomes.
"""

                retry_result, retry_input, retry_output = call_claude(
                    client, enhanced_prompt, budget_tracker, "tailoring_retry"
                )

                retry_tailored = retry_result.get('tailored_content', {})
                retry_cost = estimate_cost(retry_input, retry_output, model="sonnet")
                total_cost += retry_cost

                # Regenerate DOCX
                generate_docx_resume(resume, retry_tailored, str(docx_path))
                print(f"  Regenerated: {docx_path}")

                pdf_path = convert_to_pdf(str(docx_path))
                if pdf_path:
                    print(f"  Regenerated: {pdf_path}")

                # Update tailored_content for saving
                tailored_content = retry_tailored

            # Save quality review
            quality_path = job_output_dir / "quality_review.json"
            save_json({
                'overall_score': quality_review.overall_score,
                'relevance': quality_review.relevance,
                'impact': quality_review.impact,
                'clarity': quality_review.clarity,
                'specificity': quality_review.specificity,
                'improvements': quality_review.improvements,
                'passes_threshold': quality_review.passes_threshold,
                'regenerated': regenerated,
            }, str(quality_path))

        except Exception as e:
            print(f"  Warning: Quality review failed: {e}")

    # Save tailoring results
    tailored_path = job_output_dir / "tailored_content.json"
    tailored_data = {
        'strategy': strategy,
        'summary': tailored_content.get('summary', ''),
        'skills_order': tailored_content.get('skills_order', []),
        'skills_to_add': tailored_content.get('skills_to_add', []),
        'experiences': tailored_content.get('experiences', []),
    }
    if variant_selection and variant_selection.variant:
        tailored_data['variant'] = {
            'name': variant_selection.variant,
            'method': variant_selection.selection_method,
        }
    else:
        tailored_data['variant'] = None
    tailored_data['render_data'] = build_render_data(resume, tailored_content)
    save_json(tailored_data, str(tailored_path))

    # Save analysis (include job URL for reference)
    analysis_path = job_output_dir / "analysis.json"
    analysis_with_url = {**analysis, 'job_url': job_url}
    save_json(analysis_with_url, str(analysis_path))

    # Save tailoring debug info
    save_model_inputs(
        job_id=f"{job_slug}-tailoring",
        prompt=tailoring_prompt,
        response=tailoring_result,
        output_dir=str(job_output_dir),
        input_tokens=tailor_input,
        output_tokens=tailor_output,
        cost=tailor_cost
    )

    # =========================================================================
    # Step 5: Outreach generation (API call 2) - optional
    # =========================================================================
    outreach_targets = []
    outreach_scripts = {}
    outreach_cost = 0.0

    if not skip_outreach:
        # Search for profiles
        profiles = []
        if not skip_profiles:
            outreach_context = resume.get('outreach_context', {})
            if outreach_context:
                print(f"  Searching for LinkedIn profiles at {company}...")
                profiles = search_linkedin_profiles(client, company, outreach_context, budget_tracker)
                print(f"  Found {len(profiles)} potential contacts")

        if profiles:
            # Build and send outreach prompt
            print(f"  Generating outreach scripts...")
            candidate_summary = resume.get('summary', {}).get('default', '')
            candidate_strengths = analysis.get('candidate_strengths', [])

            outreach_prompt = build_outreach_prompt(
                company=company,
                role=role,
                candidate_summary=candidate_summary,
                candidate_strengths=candidate_strengths,
                profiles=profiles,
                outreach_context=resume.get('outreach_context', {})
            )

            outreach_result, outreach_input, outreach_output = call_claude(
                client, outreach_prompt, budget_tracker, "outreach"
            )

            outreach_targets = outreach_result.get('outreach_targets', [])
            outreach_scripts = outreach_result.get('outreach_scripts', {})
            outreach_cost = estimate_cost(outreach_input, outreach_output, model="sonnet")
            total_cost += outreach_cost

            # Save outreach results
            if outreach_targets:
                outreach_md = render_outreach_md(outreach_targets, outreach_scripts, company, role)
                outreach_path = job_output_dir / "outreach.md"
                outreach_path.write_text(outreach_md, encoding='utf-8')
                print(f"  Generated: {outreach_path}")

                outreach_json_path = job_output_dir / "outreach.json"
                save_json({
                    'targets': outreach_targets,
                    'scripts': outreach_scripts
                }, str(outreach_json_path))

            # Save outreach debug info
            save_model_inputs(
                job_id=f"{job_slug}-outreach",
                prompt=outreach_prompt,
                response=outreach_result,
                output_dir=str(job_output_dir),
                input_tokens=outreach_input,
                output_tokens=outreach_output,
                cost=outreach_cost
            )

    # =========================================================================
    # Return summary
    # =========================================================================
    return {
        'slug': job_slug,
        'job_url': job_url,
        'company': company,
        'role': role,
        'analysis': analysis,
        'strategy': strategy,
        'outreach_targets': outreach_targets,
        'output_dir': str(job_output_dir),
        'cost': total_cost,
        'cost_breakdown': {
            'tailoring': tailor_cost,
            'outreach': outreach_cost,
        },
        'variant': {
            'name': variant_selection.variant if variant_selection else None,
            'method': variant_selection.selection_method if variant_selection else 'yaml_fallback',
            'ambiguity': variant_selection.ambiguity_level if variant_selection else None,
        } if variant_selection else None,
        'quality_review': {
            'score': quality_review.overall_score if quality_review else None,
            'passed': quality_review.passes_threshold if quality_review else None,
            'regenerated': regenerated,
        } if quality_review else None,
    }


# =============================================================================
# Utilities
# =============================================================================

def extract_company_name(job_description: str) -> str:
    """Extract company name from job description using heuristics."""
    patterns = [
        r"(?:at|@|join)\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)?)",
        r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)?)\s+is\s+(?:hiring|looking|seeking)",
        r"^([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)?)\s*[-|]",
        r"About\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, job_description)
        if match:
            return match.group(1).strip()

    return "Unknown"


def fetch_job_from_url(url: str, allow_linkedin_fallback: bool = True) -> str:
    """
    Fetch job description from a URL using web search.

    If direct fetch fails and allow_linkedin_fallback is True, attempts to find
    the same job on LinkedIn.

    Raises:
        ValueError: If the page content couldn't be fetched or doesn't look like a job posting
    """
    client = anthropic.Anthropic()

    # First, try direct fetch
    text_content, fetch_failed, fail_reason = _try_fetch_url(client, url)

    if not fetch_failed:
        return text_content

    # Direct fetch failed - try LinkedIn fallback
    if allow_linkedin_fallback:
        print(f"  Direct fetch failed: {fail_reason}")
        print(f"  Trying LinkedIn fallback...")

        linkedin_content = _try_linkedin_fallback(client, url, fail_reason)
        if linkedin_content:
            return linkedin_content

    # Both failed
    raise ValueError(f"{fail_reason}. Try using --job-file with pasted content instead.")


def _try_fetch_url(client: anthropic.Anthropic, url: str) -> tuple[str, bool, str]:
    """
    Attempt to fetch job description from URL.

    Returns:
        Tuple of (content, failed, reason)
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{
            "role": "user",
            "content": f"""Fetch the job posting from this URL and extract the full job description:
{url}

Return ONLY the job description text, including:
- Job title
- Company name
- Location
- Requirements
- Responsibilities
- Qualifications
- Benefits (if listed)

Do not add commentary, just return the job posting content.

If you cannot access the page or it requires login, respond with exactly: FETCH_FAILED: <reason>"""
        }]
    )

    text_content = ""
    for block in response.content:
        if hasattr(block, 'text'):
            text_content += block.text

    # Check for explicit fetch failure
    if text_content.strip().startswith("FETCH_FAILED:"):
        reason = text_content.replace("FETCH_FAILED:", "").strip()
        return "", True, f"Could not fetch job posting: {reason}"

    # Check for signs of failed fetch
    failure_indicators = [
        "unable to access",
        "cannot access",
        "couldn't access",
        "could not access",
        "login required",
        "sign in required",
        "authentication required",
        "page not found",
        "404",
        "access denied",
        "forbidden",
        "blocked",
        "captcha",
        "verify you are human",
        "job posting unavailable",
        "job not found",
        "position has been filled",
        "no longer available",
        "this job is no longer",
        "posting is no longer",
        "unable to retrieve",
        "could not retrieve",
        "cannot retrieve",
        "not accessible",
        "requires login",
        "requires authentication",
    ]

    text_lower = text_content.lower()
    for indicator in failure_indicators:
        if indicator in text_lower:
            return "", True, f"Page inaccessible (detected: '{indicator}')"

    # Check minimum content length
    if len(text_content.strip()) < 800:
        return "", True, f"Fetched content too short ({len(text_content)} chars)"

    return text_content, False, ""


def _try_linkedin_fallback(client: anthropic.Anthropic, original_url: str, fail_reason: str) -> Optional[str]:
    """
    Try to find the same job on LinkedIn when direct URL fails.

    Extracts company name from URL and searches LinkedIn Jobs.

    Returns:
        Job description if found, None otherwise
    """
    # Extract company name from URL
    company_hints = _extract_company_from_url(original_url)

    if not company_hints:
        print(f"  Could not determine company from URL for LinkedIn search")
        return None

    # Search LinkedIn for the job
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{
            "role": "user",
            "content": f"""I'm trying to find a job posting from {company_hints['company']} that I couldn't access directly.

Original URL: {original_url}
{f"Job ID hint: {company_hints['job_id']}" if company_hints.get('job_id') else ""}

Search LinkedIn Jobs for this position at {company_hints['company']}.
Look for: site:linkedin.com/jobs {company_hints['company']}

Find the matching job posting and extract the FULL job description including:
- Job title
- Company name
- Location
- Requirements
- Responsibilities
- Qualifications
- Benefits (if listed)

Return ONLY the job description text. No commentary.

If you cannot find a matching job, respond with exactly: LINKEDIN_NOT_FOUND"""
        }]
    )

    text_content = ""
    for block in response.content:
        if hasattr(block, 'text'):
            text_content += block.text

    # Check if LinkedIn search failed
    if "LINKEDIN_NOT_FOUND" in text_content:
        print(f"  LinkedIn fallback: Could not find matching job")
        return None

    # Validate the content
    if len(text_content.strip()) < 800:
        print(f"  LinkedIn fallback: Content too short ({len(text_content)} chars)")
        return None

    print(f"  LinkedIn fallback: Found job ({len(text_content)} chars)")
    return text_content


def _extract_company_from_url(url: str) -> Optional[dict]:
    """
    Extract company name and job ID from career page URLs.

    Supports common ATS patterns:
    - metacareers.com, meta.com -> Meta
    - greenhouse.io/company/... -> company
    - lever.co/company/... -> company
    - jobs.company.com -> company
    - company.com/careers/... -> company
    """
    url_lower = url.lower()

    # Meta/Facebook
    if 'metacareers.com' in url_lower or 'meta.com/careers' in url_lower:
        job_id_match = re.search(r'/(\d+)', url)
        return {'company': 'Meta', 'job_id': job_id_match.group(1) if job_id_match else None}

    # Google
    if 'careers.google.com' in url_lower or 'google.com/careers' in url_lower:
        return {'company': 'Google', 'job_id': None}

    # Apple
    if 'jobs.apple.com' in url_lower or 'apple.com/careers' in url_lower:
        return {'company': 'Apple', 'job_id': None}

    # Amazon
    if 'amazon.jobs' in url_lower or 'amazon.com/jobs' in url_lower:
        return {'company': 'Amazon', 'job_id': None}

    # Microsoft
    if 'careers.microsoft.com' in url_lower:
        return {'company': 'Microsoft', 'job_id': None}

    # Greenhouse pattern: boards.greenhouse.io/company/...
    greenhouse_match = re.search(r'greenhouse\.io/([^/]+)', url_lower)
    if greenhouse_match:
        company = greenhouse_match.group(1).replace('-', ' ').title()
        return {'company': company, 'job_id': None}

    # Lever pattern: jobs.lever.co/company/...
    lever_match = re.search(r'lever\.co/([^/]+)', url_lower)
    if lever_match:
        company = lever_match.group(1).replace('-', ' ').title()
        return {'company': company, 'job_id': None}

    # Generic: jobs.company.com or company.com/careers
    generic_match = re.search(r'(?:jobs\.|careers\.)?([a-z0-9-]+)\.(?:com|io|co|org)', url_lower)
    if generic_match:
        company = generic_match.group(1).replace('-', ' ').title()
        # Skip common non-company domains
        if company.lower() not in ['www', 'linkedin', 'indeed', 'glassdoor', 'ziprecruiter']:
            return {'company': company, 'job_id': None}

    return None
