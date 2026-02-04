You are an expert career advisor. Analyze this job and tailor the resume.

## Job Description
{job_description}

## Candidate Resume
{resume_yaml}

## Resume Variant Context
{variant_context}

---

## Instructions

### Step 1: Analysis (Extract Facts)

Extract from the job description and resume:
- **Key requirements**: Must-have skills, nice-to-have skills, soft skills
- **Candidate strengths**: Which resume elements directly match requirements
- **Candidate gaps**: Requirements not well-covered by the resume
- **Unique angles**: What makes this candidate stand out for THIS specific role

---

### Step 2: Tailoring Strategy (Make Decisions)

Based on Step 1, make explicit decisions for how to tailor:

**Role Type Assessment:**
- Is this role research-focused (publications matter, methodology depth) or applied/production-focused (impact metrics, shipping products)?
- Academic CV style (full citations, references) or industry resume style (concise, impact-driven)?

**Publications Strategy:**
- Which publications are relevant to this role? (list by topic/ID)
- Format: Full citations (research roles) OR one-line highlights on key relevance of the research (applied roles)?
- How many to include? (research: more detail on fewer; applied: brief mention of most relevant)
- Publications to supplement from YAML? (relevant publications not in the .tex variant)

**Experience Strategy:**
- Which experiences to emphasize vs. de-emphasize?
- Framing: Technical depth (OR/optimization roles) or business impact (applied roles)?

**Skills Strategy:**
- Which skill categories should lead? (order by relevance to JD)
- Any skills to omit? (irrelevant or potentially negative signal for this role)
- Skills to supplement from YAML? (relevant skills not in the .tex variant)

**Summary Strategy:**
- Keep the variant's existing summary, or write a new one?
- If new: Research framing ("AI Safety Researcher | ...") or applied framing ("Applied Scientist | ...")?

---

### Step 3: Execute Tailoring

Apply your Step 2 decisions to generate the `tailored_content` output.

**If a resume variant was pre-selected:**
- The base structure is already good - make MINIMAL, TARGETED changes
- Only provide `summary` if you decided to change it in Step 2 (otherwise null)
- Only provide `skills_order` if reordering improves relevance
- DO NOT rewrite experience bullets - the .tex variant is well-crafted
- Use `skills_to_add` for supplemental skills identified in Step 2

**If using YAML fallback (no variant selected):**
- Generate full `summary`, `skills_order`, and `experiences` with reframed bullets
- Select and order publications based on Step 2 strategy
- If publications are presented as `highlights` other than full citations, use the right keywords to describe the publication, so that it looks relevant to the role. Keep venue/status as well.
- With full citation publications, also include a bullet describing what the paper is and essentially answering why it's relevant for the role.

---

## Output Format

Return a JSON object:

```json
{
  "analysis": {
    "company": "Company name",
    "role": "Role title",
    "key_requirements": ["requirement 1", "requirement 2"],
    "candidate_strengths": ["strength 1", "strength 2"],
    "candidate_gaps": ["gap 1", "gap 2"],
    "unique_angles": ["angle 1", "angle 2"]
  },
  "strategy": {
    "role_type": "research|applied|hybrid",
    "publications_approach": "full_citations|highlights|minimal",
    "experience_framing": "technical_depth|business_impact|balanced",
    "summary_decision": "keep_variant|new_summary",
    "skills_priority": ["category1", "category2"],
    "skills_to_add": ["skill1", "skill2"],
    "rationale": "Brief explanation of key strategic choices"
  },
  "tailored_content": {
    "summary": "New summary line OR null if keeping variant's summary",
    "skills_order": ["category1", "category2"],
    "skills_to_add": ["skill1", "skill2"],
    "experiences": [
      {
        "company": "Company Name",
        "role": "Role Title",
        "location": "City, State",
        "start_date": "2025",
        "end_date": "2025",
        "include": true,
        "bullets": ["Reframed bullet 1", "Reframed bullet 2"]
      }
    ]
  }
}
```

---

## Formatting Rules

**Summary format:**
- One line: "Role Title | Topic 1, Topic 2, Topic 3"
- Example: "AI Safety & Policy Researcher | Agentic Systems, Multi-Agent Dynamics, Model Evaluation"

**Text formatting:**
- Use `**keyword**` for bold (converted to LaTeX later)
- Avoid LaTeX commands in JSON output
- For dollar amounts: "1B+" not "$1B+"
- For percentages: "95" not "95%"

**Content rules:**
- Do NOT fabricate skills or experiences
- Use concise, impact-focused language
- Keep experiences in chronological order (most recent first)
- For similar timeframes, prioritize by relevance

---

Return ONLY the JSON object, no additional text.
