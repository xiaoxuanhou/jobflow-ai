You are ranking job opportunities for a candidate. You will see ALL viable jobs at once and must rank them from most to least relevant.

## Candidate Profile
{resume_summary}

## Candidate Preferences
Target roles: {target_roles}
Key interests: {key_interests}
Preferred: {preferences}
Avoid: {avoid}

## Ranking Criteria (weights)
{ranking_weights}

## Jobs to Rank
{all_jobs}

## Task
Rank ALL jobs from #1 (best fit) to #N (worst fit).

Guidelines:
- Be decisive — avoid ties
- Consider the WHOLE pool when ranking (relative comparison)
- A job can be good but ranked low if others are better
- Focus on how well each role matches the candidate's background AND interests

Return JSON:
{
  "rankings": [
    {"rank": 1, "job_id": "...", "company": "...", "role": "...", "justification": "Why this is #1"},
    {"rank": 2, "job_id": "...", "company": "...", "role": "...", "justification": "Why this is #2"},
    ...
  ],
  "top_5_summary": "Brief explanation of why the top 5 stand out from the rest"
}

Return ONLY the JSON object, no additional text.
