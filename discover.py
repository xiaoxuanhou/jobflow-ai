"""
Job discovery module for jobflow-ai.

Finds new job postings from target companies using web search.
"""

import hashlib
import json
import re
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import anthropic

from budget import BudgetTracker, estimate_cost, COSTS


def discover_jobs(
    companies: list[dict],
    criteria: dict,
    budget_tracker: Optional[BudgetTracker] = None
) -> list[dict]:
    """
    Discover new job postings from target companies.

    Args:
        companies: List of company dicts with name, careers_url, ats
        criteria: Search criteria dict
        budget_tracker: Optional budget tracker

    Returns:
        List of job posting dicts
    """
    client = anthropic.Anthropic()
    all_jobs = []

    # Get target role titles
    titles = criteria.get('search', {}).get('titles', ['Software Engineer'])

    for company in companies:
        company_name = company.get('name', 'Unknown')
        careers_url = company.get('careers_url', '')

        print(f"  Searching: {company_name}...")

        # Search for jobs at this company
        jobs = search_company_careers(
            client, company_name, careers_url, titles, budget_tracker
        )
        all_jobs.extend(jobs)

        # Track cost
        if budget_tracker:
            budget_tracker.add("discovery", COSTS['discovery_per_company'], {
                "company": company_name,
                "jobs_found": len(jobs)
            })

    # Also run general search if enabled
    linkedin_config = criteria.get('linkedin_search', {})
    if linkedin_config.get('enabled', False):
        queries = linkedin_config.get('queries', [])
        locations = linkedin_config.get('locations', [])

        for query in queries[:3]:  # Limit queries
            print(f"  Searching: {query}...")
            jobs = search_linkedin_jobs(client, query, locations, budget_tracker)
            all_jobs.extend(jobs)

    return all_jobs


def search_company_careers(
    client: anthropic.Anthropic,
    company_name: str,
    careers_url: str,
    titles: list[str],
    budget_tracker: Optional[BudgetTracker] = None
) -> list[dict]:
    """
    Search for jobs at a specific company.

    First tries the company's careers page. If that returns no jobs (page
    inaccessible, blocked, etc.), falls back to searching LinkedIn for
    jobs at that company.

    Args:
        client: Anthropic client
        company_name: Company name
        careers_url: Company careers page URL
        titles: List of target role titles
        budget_tracker: Optional budget tracker

    Returns:
        List of job dicts
    """
    titles_str = ", ".join(titles[:5])  # Limit titles in query

    # Step 1: Try company careers page
    jobs = _search_careers_page(client, company_name, careers_url, titles_str)

    if jobs:
        print(f"    Found {len(jobs)} jobs from careers page")
        return jobs

    # Step 2: Careers page returned nothing - try LinkedIn fallback
    print(f"    Careers page returned 0 jobs, trying LinkedIn...")
    jobs = _search_linkedin_for_company(client, company_name, titles_str, budget_tracker)

    if jobs:
        print(f"    Found {len(jobs)} jobs from LinkedIn")
    else:
        print(f"    No jobs found for {company_name}")

    return jobs


def _search_careers_page(
    client: anthropic.Anthropic,
    company_name: str,
    careers_url: str,
    titles_str: str
) -> list[dict]:
    """Try to search company's careers page directly."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": f"""Search for recent job postings at {company_name} for these roles: {titles_str}

Look at their careers page: {careers_url}

Return a JSON array of job postings found. For each job include:
- title: Job title
- company: Company name
- url: Direct URL to the job posting
- location: Job location
- posted_date: When posted (if available, estimate if within 24h/week/month)
- description_preview: First 200 chars of description

Only include jobs that appear to be posted within the last 7 days.
Return ONLY the JSON array. If no jobs found or page inaccessible, return [].

Example format:
[
  {{
    "title": "Senior Software Engineer",
    "company": "{company_name}",
    "url": "https://...",
    "location": "San Francisco, CA",
    "posted_date": "2025-01-28",
    "description_preview": "We are looking for..."
  }}
]"""
            }]
        )

        text_content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text_content += block.text

        return parse_jobs_from_response(text_content, company_name)

    except Exception as e:
        print(f"    Warning: Careers page search failed: {e}")
        return []


def _search_linkedin_for_company(
    client: anthropic.Anthropic,
    company_name: str,
    titles_str: str,
    budget_tracker: Optional[BudgetTracker] = None
) -> list[dict]:
    """Search LinkedIn for jobs at a specific company (fallback)."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": f"""Search LinkedIn Jobs for positions at {company_name}.

Target roles: {titles_str}

Search query: site:linkedin.com/jobs "{company_name}" ({titles_str.replace(', ', ' OR ')})

Return a JSON array of job postings found. For each job include:
- title: Job title
- company: Company name (should be {company_name})
- url: LinkedIn job URL
- location: Job location
- posted_date: When posted (if available)
- description_preview: First 200 chars of description

Only include jobs that appear to be posted within the last 7 days.
Return ONLY the JSON array. If no jobs found, return [].

Example format:
[
  {{
    "title": "Senior Software Engineer",
    "company": "{company_name}",
    "url": "https://linkedin.com/jobs/...",
    "location": "San Francisco, CA",
    "posted_date": "2025-01-28",
    "description_preview": "We are looking for..."
  }}
]"""
            }]
        )

        text_content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text_content += block.text

        # Track additional cost for LinkedIn fallback
        if budget_tracker:
            cost = estimate_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
                model="sonnet"
            )
            budget_tracker.add("linkedin_company_fallback", cost, {"company": company_name})

        return parse_jobs_from_response(text_content, company_name)

    except Exception as e:
        print(f"    Warning: LinkedIn fallback search failed: {e}")
        return []


def search_linkedin_jobs(
    client: anthropic.Anthropic,
    query: str,
    locations: list[str],
    budget_tracker: Optional[BudgetTracker] = None
) -> list[dict]:
    """
    Search for jobs using general web search (LinkedIn, Google Jobs, etc.).

    Args:
        client: Anthropic client
        query: Search query
        locations: List of target locations
        budget_tracker: Optional budget tracker

    Returns:
        List of job dicts
    """
    locations_str = ", ".join(locations[:3]) if locations else "Remote"

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": f"""Search for recent job postings matching: {query}

Target locations: {locations_str}

Search LinkedIn Jobs, Google Jobs, or company career sites.

Return a JSON array of job postings. For each job include:
- title: Job title
- company: Company name
- url: Direct URL to the job posting
- location: Job location
- posted_date: When posted (if available)
- description_preview: First 200 chars of description

Only include jobs posted within the last 7 days.
Return ONLY the JSON array. If no jobs found, return []."""
            }]
        )

        # Extract text content
        text_content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text_content += block.text

        # Parse JSON
        jobs = parse_jobs_from_response(text_content, None)

        # Track cost
        if budget_tracker:
            cost = estimate_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
                model="sonnet"
            )
            budget_tracker.add("linkedin_search", cost, {"query": query})

        return jobs

    except Exception as e:
        print(f"    Warning: LinkedIn search failed: {e}")
        return []


def parse_jobs_from_response(text: str, default_company: Optional[str]) -> list[dict]:
    """Parse job postings from API response text."""
    jobs = []

    # Find JSON array in response
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return jobs

    try:
        parsed = json.loads(match.group())
        for job in parsed:
            if isinstance(job, dict) and job.get('url'):
                # Ensure company is set
                if not job.get('company') and default_company:
                    job['company'] = default_company

                # Add job ID (hash of URL)
                job['id'] = hashlib.md5(job['url'].encode()).hexdigest()[:12]

                jobs.append(job)
    except json.JSONDecodeError:
        pass

    return jobs


def fetch_job_details(
    client: anthropic.Anthropic,
    job: dict,
    budget_tracker: Optional[BudgetTracker] = None
) -> dict:
    """
    Fetch full job description from URL.

    If direct fetch fails, falls back to searching LinkedIn for the same job.

    Args:
        client: Anthropic client
        job: Job dict with url, company, title
        budget_tracker: Optional budget tracker

    Returns:
        Job dict with full description
    """
    url = job.get('url', '')
    if not url:
        return job

    company = job.get('company', 'Unknown')
    title = job.get('title', '')

    # First try direct fetch
    text_content, success = _try_direct_fetch(client, url, budget_tracker, job.get('id'))

    if success and len(text_content) >= 500:
        job['description'] = text_content
        job['description_fetched'] = True
        job['fetch_source'] = 'direct'
        return job

    # Direct fetch failed or too short - try LinkedIn fallback
    print(f"    Direct fetch insufficient, trying LinkedIn...")
    linkedin_content = _try_linkedin_job_search(client, company, title, budget_tracker, job.get('id'))

    if linkedin_content and len(linkedin_content) >= 500:
        job['description'] = linkedin_content
        job['description_fetched'] = True
        job['fetch_source'] = 'linkedin'
        return job

    # Both failed - use preview
    print(f"    Warning: Could not fetch full job details")
    job['description'] = job.get('description_preview', '')
    job['description_fetched'] = False
    job['fetch_source'] = 'preview_only'

    return job


def _try_direct_fetch(
    client: anthropic.Anthropic,
    url: str,
    budget_tracker: Optional[BudgetTracker],
    job_id: str
) -> tuple[str, bool]:
    """Try to fetch job description directly from URL."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": f"""Fetch the full job posting from this URL:
{url}

Extract and return the complete job description including:
- Full job title
- Company name
- Location
- About the role
- Responsibilities
- Requirements / Qualifications
- Nice to haves
- Benefits
- Team info

Return the full text content. Do not summarize.

If you cannot access the page, respond with: FETCH_FAILED"""
            }]
        )

        text_content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text_content += block.text

        # Track cost
        if budget_tracker:
            cost = estimate_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
                model="sonnet"
            )
            budget_tracker.add("fetch_job", cost, {"job_id": job_id})

        # Check for failure
        if "FETCH_FAILED" in text_content:
            return "", False

        # Check for common failure patterns
        failure_patterns = [
            "unable to access", "cannot access", "login required",
            "page not found", "access denied", "job posting unavailable"
        ]
        text_lower = text_content.lower()
        for pattern in failure_patterns:
            if pattern in text_lower and len(text_content) < 600:
                return "", False

        return text_content, True

    except Exception as e:
        print(f"    Warning: Direct fetch error: {e}")
        return "", False


def _try_linkedin_job_search(
    client: anthropic.Anthropic,
    company: str,
    title: str,
    budget_tracker: Optional[BudgetTracker],
    job_id: str
) -> Optional[str]:
    """Search LinkedIn for the same job as fallback."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": f"""Search LinkedIn Jobs for this position:
Company: {company}
Role: {title}

Search query: site:linkedin.com/jobs "{company}" "{title}"

Find the matching job posting and extract the FULL job description including:
- Full job title
- Company name
- Location
- About the role
- Responsibilities
- Requirements / Qualifications
- Nice to haves
- Benefits

Return the full text content. Do not summarize.

If you cannot find a matching job, respond with: LINKEDIN_NOT_FOUND"""
            }]
        )

        text_content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text_content += block.text

        # Track cost
        if budget_tracker:
            cost = estimate_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
                model="sonnet"
            )
            budget_tracker.add("linkedin_fallback", cost, {"job_id": job_id})

        if "LINKEDIN_NOT_FOUND" in text_content:
            return None

        return text_content

    except Exception as e:
        print(f"    Warning: LinkedIn fallback error: {e}")
        return None


def deduplicate_jobs(jobs: list[dict], cache_dir: str = "cache") -> list[dict]:
    """
    Remove jobs that have already been seen.

    Args:
        jobs: List of job dicts
        cache_dir: Directory for cache file

    Returns:
        List of new jobs (not in cache)
    """
    cache_path = Path(cache_dir) / "seen_jobs.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    if cache_path.exists():
        seen = json.loads(cache_path.read_text())
    else:
        seen = {}

    # Filter to new jobs
    new_jobs = []
    for job in jobs:
        job_id = job.get('id', '')
        url_hash = hashlib.md5(job.get('url', '').encode()).hexdigest()

        if job_id not in seen and url_hash not in seen:
            new_jobs.append(job)
            # Add to seen cache
            seen[job_id] = {
                'url': job.get('url', ''),
                'company': job.get('company', ''),
                'role': job.get('title', ''),
                'first_seen': datetime.now().isoformat()
            }

    # Save updated cache
    cache_path.write_text(json.dumps(seen, indent=2))

    return new_jobs


def save_discovery_log(jobs: list, rejected: list, output_dir: str = "outputs") -> Path:
    """
    Save discovery log.

    Args:
        jobs: All discovered jobs
        rejected: Jobs rejected by rule filter
        output_dir: Output directory

    Returns:
        Path to log file
    """
    today = date.today().isoformat()
    log_path = Path(output_dir) / today / "discovery_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log = {
        'date': today,
        'discovered': len(jobs),
        'jobs': jobs,
        'rule_filtered_out': len(rejected),
        'rejections': rejected
    }

    log_path.write_text(json.dumps(log, indent=2))
    return log_path
