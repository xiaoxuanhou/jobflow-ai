Quickly score these job postings for basic fit. Be generous — we'll rank properly later.

## Candidate Summary
{resume_summary}

## Criteria
Target roles: {target_roles}
Bonus if contains: {must_have}

## Jobs to Score
{jobs_batch}

Score each job 1-10 based on basic relevance:
- 1-4: Clear mismatch (wrong domain, wrong level, missing key requirements)
- 5-6: Possibly relevant (some alignment, worth a closer look)
- 7-10: Good fit (strong alignment with candidate background)

Keep threshold LOW (5+) — we just want to remove obvious mismatches.
The ranking stage will do proper comparison.

Return a JSON array with one entry per job:
[
  {"job_id": "...", "score": 6, "reason": "Brief reason for score"},
  {"job_id": "...", "score": 3, "reason": "Brief reason for score"},
  ...
]

Return ONLY the JSON array, no additional text.
