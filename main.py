#!/usr/bin/env python3
"""
jobflow-ai: AI-powered job search automation

Discovers new job postings, tailors resumes, and generates LinkedIn outreach plans.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import anthropic

from budget import BudgetTracker, BudgetExceededError, COSTS
from utils import load_yaml, save_json, generate_daily_summary


def cmd_process(args):
    """Process a single job (manual input)."""
    from process import process_job, fetch_job_from_url

    # Load resume
    resume_path = Path(args.resume)
    if not resume_path.exists():
        print(f"Error: Resume file not found: {resume_path}")
        sys.exit(1)

    resume = load_yaml(str(resume_path))

    # Get job description
    if args.job_url:
        print(f"Fetching job from URL: {args.job_url}")
        try:
            job_description = fetch_job_from_url(args.job_url)
            print(f"  Fetched {len(job_description)} characters")
        except ValueError as e:
            print(f"\nError: {e}")
            print("\nAlternative: Copy the job description and save it to a file, then run:")
            print(f"  .\\run.ps1 process --job-file <path-to-file>")
            sys.exit(1)
        job_url = args.job_url
    elif args.job_file:
        job_file = Path(args.job_file)
        if not job_file.exists():
            print(f"Error: Job file not found: {job_file}")
            sys.exit(1)
        job_description = job_file.read_text(encoding='utf-8')
        job_url = f"file://{job_file.absolute()}"
    else:
        print("Error: Must provide --job-url or --job-file")
        sys.exit(1)

    # Initialize budget tracker
    max_cost = args.max_cost if args.max_cost else float('inf')
    tracker = BudgetTracker(max_budget=max_cost)

    # Check budget before processing
    estimated_cost = COSTS['process_per_job']
    if not tracker.can_afford("process", estimated_cost):
        print(f"Error: Cannot afford processing (need ${estimated_cost:.2f}, budget ${max_cost:.2f})")
        sys.exit(1)

    try:
        print(f"\nProcessing job...")
        result = process_job(
            job_description=job_description,
            job_url=job_url,
            resume=resume,
            budget_tracker=tracker,
            output_dir=args.output,
            skip_profiles=args.skip_profiles
        )

        # Add processing cost
        tracker.add("process", result['cost'], {
            "job": result['slug'],
            "company": result['company'],
            "role": result['role']
        })

        print(f"\nProcessed: {result['role']} @ {result['company']}")
        print(f"Output: {result['output_dir']}")
        print(f"Cost: ${result['cost']:.4f}")
        print(tracker.summary())

        # Save budget log
        tracker.save_log(args.output)

        # Generate summary
        generate_daily_summary([result], args.output)

    except BudgetExceededError as e:
        print(f"\nBudget exceeded: {e}")
        tracker.save_log(args.output)
        sys.exit(1)


def cmd_discover(args):
    """Discover new jobs and optionally process them."""
    from discover import discover_jobs, deduplicate_jobs
    from filter import apply_rule_filter, filter_jobs_loose
    from rank import rank_jobs_consistently

    # Load configuration files
    criteria_path = Path(args.criteria)
    if not criteria_path.exists():
        print(f"Error: Criteria file not found: {criteria_path}")
        sys.exit(1)
    criteria = load_yaml(str(criteria_path))

    companies_path = Path(args.companies)
    if not companies_path.exists():
        print(f"Error: Companies file not found: {companies_path}")
        sys.exit(1)
    companies = load_yaml(str(companies_path))

    resume_path = Path(args.resume)
    if not resume_path.exists():
        print(f"Error: Resume file not found: {resume_path}")
        sys.exit(1)
    resume = load_yaml(str(resume_path))

    # Initialize budget tracker
    max_cost = args.max_cost if args.max_cost else float('inf')
    tracker = BudgetTracker(max_budget=max_cost)

    if args.dry_run:
        print("DRY RUN - No API calls will be made\n")

    try:
        # Stage 1: Discovery
        print("Stage 1: Discovering jobs...")
        company_list = companies.get('companies', [])

        # Optionally limit companies (for testing)
        if args.limit_companies:
            company_list = company_list[:args.limit_companies]
            print(f"  (Limited to {args.limit_companies} companies for testing)")

        num_companies = len(company_list)

        if args.dry_run:
            print(f"  Would search {num_companies} companies")
            print(f"  Estimated cost: ${num_companies * COSTS['discovery_per_company']:.2f}")
        else:
            jobs = discover_jobs(company_list, criteria, tracker)
            jobs = deduplicate_jobs(jobs, args.cache)
            print(f"  Found {len(jobs)} new jobs")

        # Stage 2: Rule-based filter (free)
        print("\nStage 2: Rule-based filtering...")
        if args.dry_run:
            print("  Would apply rule-based filters (free)")
        else:
            if not jobs:
                print("  No jobs to filter - skipping remaining stages")
                tracker.save_log(args.output)
                return

            passed_jobs, rejected = apply_rule_filter(jobs, criteria)
            tracker.add("rule_filter", 0.0, {"passed": len(passed_jobs), "rejected": len(rejected)})
            print(f"  Passed: {len(passed_jobs)}, Rejected: {len(rejected)}")

        # Stage 3: Loose AI filter
        print("\nStage 3: AI filtering (Haiku)...")
        filter_cost = COSTS['filter_batch']
        if args.dry_run:
            print(f"  Estimated cost: ${filter_cost:.2f}")
        else:
            if not passed_jobs:
                print("  No jobs passed rule filter - skipping remaining stages")
                tracker.save_log(args.output)
                return

            if not tracker.can_afford("ai_filter", filter_cost):
                tracker.abort(f"Cannot afford AI filter (need ${filter_cost:.2f})")

            scored_jobs = filter_jobs_loose(passed_jobs, resume, criteria, tracker)
            viable_jobs = [j for j in scored_jobs if j['score'] >= criteria.get('thresholds', {}).get('loose_filter_min', 5)]
            print(f"  Viable jobs (score >= 5): {len(viable_jobs)}")

        # Stage 4: Consistent ranking
        print("\nStage 4: Ranking (Sonnet)...")
        ranking_cost = COSTS['ranking']
        if args.dry_run:
            print(f"  Estimated cost: ${ranking_cost:.2f}")
        else:
            if not viable_jobs:
                print("  No viable jobs to rank - skipping remaining stages")
                tracker.save_log(args.output)
                return

            if not tracker.can_afford("ranking", ranking_cost):
                tracker.abort(f"Cannot afford ranking (need ${ranking_cost:.2f})")

            ranked_jobs = rank_jobs_consistently(viable_jobs, resume, criteria, tracker)
            print(f"  Ranked {len(ranked_jobs)} jobs")

            # Save ranking results
            today = date.today().isoformat()
            ranking_path = Path(args.output) / today / "ranking_results.json"
            ranking_path.parent.mkdir(parents=True, exist_ok=True)
            save_json({'rankings': ranked_jobs}, str(ranking_path))

        # Stage 5: Process top N (if requested)
        if args.process:
            max_jobs = criteria.get('thresholds', {}).get('max_jobs_to_process', 5)

            if args.dry_run:
                print(f"\nStage 5: Processing (up to {max_jobs} jobs)...")
                print(f"  Estimated cost: ${max_jobs * COSTS['process_per_job']:.2f}")
            else:
                from process import process_job
                from discover import fetch_job_details

                if not ranked_jobs:
                    print("  No jobs to process")
                    tracker.save_log(args.output)
                    return

                top_jobs = ranked_jobs[:max_jobs]
                print(f"\nStage 5: Processing top {len(top_jobs)} jobs...")

                # First, fetch full descriptions for jobs that don't have them
                print(f"  Fetching full job descriptions...")
                jobs_client = anthropic.Anthropic()
                for job in top_jobs:
                    if not job.get('description') or len(job.get('description', '')) < 500:
                        print(f"    Fetching: {job.get('title', 'Unknown')} @ {job.get('company', 'Unknown')}")
                        fetch_job_details(jobs_client, job, tracker)

                results = []
                for i, job in enumerate(top_jobs, 1):
                    process_cost = COSTS['process_per_job']
                    if not tracker.can_afford("process", process_cost):
                        print(f"  Budget limit reached. Processed {i-1} of {len(top_jobs)} jobs.")
                        break

                    # Use description or fall back to preview
                    job_desc = job.get('description') or job.get('description_preview', '')
                    if not job_desc:
                        print(f"  [{i}] Skipping {job.get('title', 'Unknown')} @ {job.get('company', 'Unknown')} - no description")
                        continue

                    print(f"  [{i}] Processing: {job.get('title', 'Unknown')} @ {job.get('company', 'Unknown')}")
                    result = process_job(
                        job_description=job_desc,
                        job_url=job.get('url', ''),
                        resume=resume,
                        budget_tracker=tracker,
                        output_dir=args.output,
                        skip_profiles=args.skip_profiles,
                        company=job.get('company'),
                        role_title=job.get('title')
                    )
                    results.append(result)
                    print(f"      Cost: ${result['cost']:.4f}")

                if results:
                    # Generate daily summary
                    summary_path = generate_daily_summary(results, args.output)
                    print(f"\nDaily summary: {summary_path}")

        # Final summary
        if args.dry_run:
            total_estimate = (
                num_companies * COSTS['discovery_per_company'] +
                filter_cost +
                ranking_cost +
                (5 * COSTS['process_per_job'] if args.process else 0)
            )
            print(f"\nTotal estimated cost: ${total_estimate:.2f}")
        else:
            print(f"\n{tracker.summary()}")
            tracker.save_log(args.output)

    except BudgetExceededError as e:
        print(f"\nBudget exceeded: {e}")
        tracker.save_log(args.output)
        sys.exit(1)


def cmd_cache(args):
    """Manage the seen jobs cache."""
    cache_path = Path(args.cache) / "seen_jobs.json"

    if args.clear:
        if cache_path.exists():
            cache_path.unlink()
            print(f"Cleared cache: {cache_path}")
        else:
            print("Cache already empty")
    elif args.list:
        if cache_path.exists():
            cache = json.loads(cache_path.read_text())
            print(f"Cached jobs: {len(cache)}")
            for url_hash, info in list(cache.items())[:10]:
                print(f"  {info.get('company', 'Unknown')}: {info.get('role', 'Unknown')}")
            if len(cache) > 10:
                print(f"  ... and {len(cache) - 10} more")
        else:
            print("Cache is empty")
    else:
        if cache_path.exists():
            cache = json.loads(cache_path.read_text())
            print(f"Cache contains {len(cache)} jobs")
        else:
            print("Cache is empty")


def cmd_cost(args):
    """Estimate costs for a run."""
    criteria_path = Path(args.criteria)
    if not criteria_path.exists():
        print(f"Error: Criteria file not found: {criteria_path}")
        sys.exit(1)
    criteria = load_yaml(str(criteria_path))

    companies_path = Path(args.companies)
    if not companies_path.exists():
        print(f"Error: Companies file not found: {companies_path}")
        sys.exit(1)
    companies = load_yaml(str(companies_path))

    num_companies = len(companies.get('companies', []))
    max_jobs = criteria.get('thresholds', {}).get('max_jobs_to_process', 5)

    # Calculate estimates
    discovery_cost = num_companies * COSTS['discovery_per_company']
    filter_cost = COSTS['filter_batch']
    ranking_cost = COSTS['ranking']
    process_cost = max_jobs * COSTS['process_per_job']
    total_cost = discovery_cost + filter_cost + ranking_cost + process_cost

    print("Cost Estimate")
    print("=" * 40)
    print(f"Companies to search: {num_companies}")
    print(f"Max jobs to process: {max_jobs}")
    print()
    print(f"Discovery:   ${discovery_cost:.2f}")
    print(f"Filtering:   ${filter_cost:.2f}")
    print(f"Ranking:     ${ranking_cost:.2f}")
    print(f"Processing:  ${process_cost:.2f}")
    print("-" * 40)
    print(f"Total:       ${total_cost:.2f}")
    print()
    print(f"Monthly (30 days): ${total_cost * 30:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="jobflow-ai: AI-powered job search automation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Process command
    process_parser = subparsers.add_parser('process', help='Process a single job')
    process_parser.add_argument('--job-url', help='URL of job posting to fetch and process')
    process_parser.add_argument('--job-file', help='Path to file containing job description')
    process_parser.add_argument('--resume', default='data/resume.yaml', help='Path to resume YAML')
    process_parser.add_argument('--template', default=None, help='Deprecated - DOCX is now generated directly')
    process_parser.add_argument('--output', default='outputs', help='Output directory')
    process_parser.add_argument('--max-cost', type=float, help='Maximum cost in USD')
    process_parser.add_argument('--skip-profiles', action='store_true', help='Skip LinkedIn profile search')

    # Discover command
    discover_parser = subparsers.add_parser('discover', help='Discover new jobs')
    discover_parser.add_argument('--process', action='store_true', help='Also process top jobs')
    discover_parser.add_argument('--dry-run', action='store_true', help='Preview without API calls')
    discover_parser.add_argument('--criteria', default='data/criteria.yaml', help='Path to criteria YAML')
    discover_parser.add_argument('--companies', default='data/companies.yaml', help='Path to companies YAML')
    discover_parser.add_argument('--resume', default='data/resume.yaml', help='Path to resume YAML')
    discover_parser.add_argument('--template', default=None, help='Deprecated - DOCX is now generated directly')
    discover_parser.add_argument('--output', default='outputs', help='Output directory')
    discover_parser.add_argument('--cache', default='cache', help='Cache directory')
    discover_parser.add_argument('--max-cost', type=float, help='Maximum cost in USD')
    discover_parser.add_argument('--skip-profiles', action='store_true', help='Skip LinkedIn profile search')
    discover_parser.add_argument('--limit-companies', type=int, help='Limit to first N companies (for testing)')

    # Cache command
    cache_parser = subparsers.add_parser('cache', help='Manage seen jobs cache')
    cache_parser.add_argument('--clear', action='store_true', help='Clear the cache')
    cache_parser.add_argument('--list', action='store_true', help='List cached jobs')
    cache_parser.add_argument('--cache', default='cache', help='Cache directory')

    # Cost command
    cost_parser = subparsers.add_parser('cost', help='Estimate costs')
    cost_parser.add_argument('--criteria', default='data/criteria.yaml', help='Path to criteria YAML')
    cost_parser.add_argument('--companies', default='data/companies.yaml', help='Path to companies YAML')

    args = parser.parse_args()

    if args.command == 'process':
        cmd_process(args)
    elif args.command == 'discover':
        cmd_discover(args)
    elif args.command == 'cache':
        cmd_cache(args)
    elif args.command == 'cost':
        cmd_cost(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
