# jobflow-ai

AI-powered job search automation: discovers new postings, tailors your resume, and generates LinkedIn outreach plans. Built with Claude API.

## Features

- **Automated Discovery**: Finds new job postings from target companies within 24 hours
- **Smart Filtering**: Three-stage filtering (rule-based → AI scoring → ranking) to surface the best matches
- **Resume Variant Selection**: Automatically selects the best resume variant for each role type
- **Resume Tailoring**: Generates customized LaTeX resumes optimized for each role
- **Outreach Generation**: Creates personalized LinkedIn connection scripts based on shared backgrounds
- **Cost-Conscious**: Estimates ~$0.50/day for 15 companies, capped at 5 jobs processed
- **GitHub Actions Ready**: Runs daily via cron, sends email alerts with results

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/jobflow-ai.git
cd jobflow-ai
pip install -r requirements.txt
```

### 2. Configure Your Resume

Copy and edit the example file:

```bash
cp data/resume.yaml.example data/resume.yaml
```

Then edit `data/resume.yaml` with your information.

**Tip**: If you have an existing resume (PDF, .tex, or Word), you can use an LLM (like Claude) to convert it to the YAML format. Just paste your resume content and ask it to convert to the schema shown in `data/resume.yaml.example`.

```yaml
contact:
  name: "Your Name"
  email: "you@example.com"
  # ...

experience:
  - company: "Your Company"
    role: "Your Role"
    highlights:
      - "Achievement 1"
      - "Achievement 2"
    tags: ["python", "ml"]

# See data/resume.yaml.example for full schema
```

### 3. Configure Target Companies

```bash
cp data/companies.yaml.example data/companies.yaml
```

Edit `data/companies.yaml`:

```yaml
companies:
  - name: "Anthropic"
    careers_url: "https://www.anthropic.com/careers"
    ats: "greenhouse"

  - name: "OpenAI"
    careers_url: "https://openai.com/careers"
    ats: "greenhouse"

  # Add your target companies...
```

### 4. Configure Search Criteria

Edit `data/criteria.yaml`:

```yaml
search:
  titles:
    - "Machine Learning Engineer"
    - "Applied Scientist"

  must_have_any:
    - "machine learning"
    - "LLM"
    - "AI"

  exclude_if:
    - "clearance required"
    - "Director"
```

### 5. (Optional) Configure Resume Variants

If you have multiple resume versions for different role types:

```bash
cp data/resume_variants/meta.yaml.example data/resume_variants/meta.yaml
```

Edit to define keyword signals that map job descriptions to your resume variants. See the example file for the schema.

### 6. Set Your API Key

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

### 7. Run

```bash
# Process a specific job manually
python main.py process --job-url "https://example.com/job"
python main.py process --job-file path/to/job.txt

# Discover new jobs and process top 5
python main.py discover --process

# Dry run (preview without API calls)
python main.py discover --process --dry-run

# With budget limit
python main.py discover --process --max-cost 0.50
```

## Usage

### Commands

```bash
# Process a single job
python main.py process --job-url "URL"
python main.py process --job-file path/to/job.txt

# Discover and process jobs
python main.py discover                    # List new jobs
python main.py discover --process          # Discover, filter, rank, process
python main.py discover --dry-run          # Preview without API calls

# Manage cache
python main.py cache                       # Show cache status
python main.py cache --clear               # Clear seen jobs
python main.py cache --list                # List cached jobs

# Cost estimation
python main.py cost                        # Estimate daily cost
```

### Options

| Option | Description |
|--------|-------------|
| `--max-cost AMOUNT` | Set budget limit in USD |
| `--skip-profiles` | Skip LinkedIn profile search |
| `--output DIR` | Output directory (default: `outputs/`) |
| `--resume FILE` | Resume YAML path (default: `data/resume.yaml`) |
| `--template FILE` | LaTeX template (default: `data/templates/resume.tex.j2`) |

## Output Structure

```
outputs/2025-01-29/
├── discovery_log.json       # All jobs found + filter rejections
├── ranking_results.json     # Full ranking with justifications
├── budget_log.json          # Cost tracking
├── daily_summary.md         # Email digest content
├── anthropic-applied-scientist/
│   ├── resume.tex           # Tailored LaTeX resume
│   ├── resume.pdf           # Compiled PDF (if pdflatex available)
│   ├── outreach.md          # LinkedIn scripts
│   ├── outreach.json        # Structured outreach data
│   ├── analysis.json        # Job analysis
│   └── debug/
│       ├── prompt.md        # Exact prompt sent to model
│       ├── response.json    # Raw response
│       └── metadata.json    # Token counts, cost
└── stripe-ml-engineer/
    └── ...
```

## Cost Estimates

| Stage | Cost | Description |
|-------|------|-------------|
| Discovery | ~$0.01/company | Web search for jobs |
| Rule Filter | $0.00 | Free deterministic filtering |
| AI Filter | ~$0.01 total | Haiku scoring (batched) |
| Ranking | ~$0.02 | Sonnet ranking (single call) |
| Processing | ~$0.06/job | Analysis + tailoring + outreach |

**Daily total** (15 companies, 5 jobs processed): **~$0.48**

**Monthly**: ~$14

## GitHub Actions Setup

1. Go to your repo → Settings → Secrets and variables → Actions

2. Add these **Secrets**:
   - `ANTHROPIC_API_KEY`: Your Claude API key
   - `EMAIL_USER`: Gmail address to send from
   - `EMAIL_APP_PASSWORD`: [Gmail app password](https://myaccount.google.com/apppasswords)
   - `EMAIL_TO`: Your email address

3. Optionally add **Variables**:
   - `DAILY_BUDGET`: Daily cost limit (default: 0.75)

4. The workflow runs daily at 9 AM EST. You can also trigger it manually from the Actions tab.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Discovery     │────▶│    Filtering    │────▶│    Ranking      │
│ (Web Search)    │     │ (Rules + Haiku) │     │ (Sonnet)        │
│ ~$0.01/company  │     │ ~$0.01 total    │     │ ~$0.02          │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Variant Select │
                                               │ (Rule + Haiku)  │
                                               └────────┬────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   Processing    │
                                               │ (Top 5 Jobs)    │
                                               │ ~$0.06/job      │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │    Outputs      │
                                               │ - Tailored .tex │
                                               │ - Outreach .md  │
                                               │ - Analysis .json│
                                               └─────────────────┘
```

## Customization

### Resume Variants

Place your `.tex` resume variants in `data/original/` and configure `data/resume_variants/meta.yaml` with keyword signals. The system will:

1. Score each variant against the job description (free, rule-based)
2. Auto-select if there's a clear winner, or use Haiku for tiebreaking (~$0.002)
3. Apply minimal tailoring (summary, skill order) to the selected variant

### LaTeX Template

Edit `data/templates/resume.tex.j2`. Uses Jinja2 with modified delimiters to avoid LaTeX conflicts:

- `<% %>` for blocks (if/for)
- `<< >>` for variables
- `<# #>` for comments

### Ranking Priorities

In `data/criteria.yaml`:

```yaml
ranking_priorities:
  skills_match: 0.25
  role_type_match: 0.25
  domain_focus: 0.20
  research_production: 0.15
  company_fit: 0.10
  seniority_fit: 0.05
```

### Outreach Context

In `data/resume.yaml`, add shared backgrounds for better outreach matching:

```yaml
outreach_context:
  schools:
    - "Stanford University"
  previous_companies:
    - "Google"
    - "Meta"
  backgrounds:
    - "Open source contributor"
```

## Contributing

Contributions welcome! Please submit PRs to the `main` branch.

## License

MIT

## Acknowledgments

Built with [Claude API](https://www.anthropic.com/api) by Anthropic.
