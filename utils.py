"""
Utility functions for jobflow-ai.

Handles LaTeX rendering, PDF compilation, file output, and debugging.
"""

import json
import re
import subprocess
import shutil
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, BaseLoader


def slugify(text: str) -> str:
    """
    Convert text to a safe folder/file name.

    Args:
        text: Text to slugify (e.g., "Applied Scientist - Anthropic")

    Returns:
        Safe slug (e.g., "applied-scientist-anthropic")
    """
    # Convert to lowercase
    text = text.lower()
    # Replace spaces and underscores with hyphens
    text = re.sub(r'[\s_]+', '-', text)
    # Remove non-alphanumeric characters (except hyphens)
    text = re.sub(r'[^a-z0-9\-]', '', text)
    # Remove multiple consecutive hyphens
    text = re.sub(r'-+', '-', text)
    # Remove leading/trailing hyphens
    text = text.strip('-')
    # Limit length
    return text[:60]


def latex_escape(text: str) -> str:
    """
    Escape special LaTeX characters in text.

    Args:
        text: Raw text that may contain special characters

    Returns:
        LaTeX-safe escaped text
    """
    if not isinstance(text, str):
        text = str(text)

    # Order matters: escape backslash first
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
    ]

    for char, escaped in replacements:
        text = text.replace(char, escaped)

    return text


def postprocess_latex(content: str) -> str:
    """
    Post-process rendered LaTeX to fix common issues.

    - Convert markdown bold **text** to LaTeX \\textbf{text}
    - Escape unescaped special characters

    Args:
        content: Raw rendered LaTeX string

    Returns:
        Fixed LaTeX string
    """
    # Convert markdown bold **text** to \textbf{text}
    content = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', content)

    # Fix unescaped & (but not already escaped \&)
    # Look for & not preceded by \
    content = re.sub(r'(?<!\\)&', r'\\&', content)

    # Fix unescaped % (but not already escaped \% or in comments)
    # This is tricky - only fix % that's not at line start (LaTeX comments)
    lines = content.split('\n')
    fixed_lines = []
    for line in lines:
        if not line.strip().startswith('%'):
            # Fix unescaped % not at start of line
            line = re.sub(r'(?<!\\)%(?!%)', r'\\%', line)
        fixed_lines.append(line)
    content = '\n'.join(fixed_lines)

    # Fix unescaped $ (but not already escaped \$)
    content = re.sub(r'(?<!\\)\$', r'\\$', content)

    return content


def render_latex(tailored_content: dict, template_path: str) -> str:
    """
    Render LaTeX resume using Jinja2 template.

    Args:
        tailored_content: Dict with tailored resume content
        template_path: Path to the .tex.j2 template

    Returns:
        Rendered LaTeX string
    """
    template_path = Path(template_path)

    try:
        env = Environment(
            loader=FileSystemLoader(str(template_path.parent)),
            # Use different delimiters to avoid LaTeX conflicts
            block_start_string='<%',
            block_end_string='%>',
            variable_start_string='<<',
            variable_end_string='>>',
            comment_start_string='<#',
            comment_end_string='#>',
            autoescape=False,
        )

        # Add custom filter for LaTeX escaping
        env.filters['latex_escape'] = latex_escape

        template = env.get_template(template_path.name)
        rendered = template.render(**tailored_content)

        # Post-process to fix markdown and special characters
        return postprocess_latex(rendered)
    except Exception as e:
        # Print more debug info
        print(f"  Debug: Template path = {template_path}")
        print(f"  Debug: Template exists = {template_path.exists()}")
        print(f"  Debug: Error type = {type(e).__name__}")
        print(f"  Debug: Error = {e}")
        raise


def render_latex_from_string(tailored_content: dict, template_string: str) -> str:
    """
    Render LaTeX resume from a template string.

    Args:
        tailored_content: Dict with tailored resume content
        template_string: Template as a string

    Returns:
        Rendered LaTeX string
    """
    env = Environment(
        loader=BaseLoader(),
        block_start_string='<%',
        block_end_string='%>',
        variable_start_string='<<',
        variable_end_string='>>',
        comment_start_string='<#',
        comment_end_string='#>',
    )
    template = env.from_string(template_string)
    return template.render(**tailored_content)


def find_latex_compiler() -> Optional[str]:
    """
    Find an available LaTeX compiler.

    Returns:
        Full path to the compiler or None
    """
    import os

    compilers = ['pdflatex', 'xelatex', 'lualatex']

    # First check if any compiler is in PATH
    for compiler in compilers:
        path = shutil.which(compiler)
        if path:
            return path

    # On Windows, check common MiKTeX installation locations
    if os.name == 'nt':
        user_home = os.path.expanduser('~')
        miktex_paths = [
            os.path.join(user_home, 'AppData', 'Local', 'Programs', 'MiKTeX', 'miktex', 'bin', 'x64'),
            os.path.join(user_home, 'AppData', 'Local', 'MiKTeX', 'miktex', 'bin', 'x64'),
            r'C:\Program Files\MiKTeX\miktex\bin\x64',
            r'C:\MiKTeX\miktex\bin\x64',
        ]

        for miktex_path in miktex_paths:
            for compiler in compilers:
                full_path = os.path.join(miktex_path, f'{compiler}.exe')
                if os.path.isfile(full_path):
                    return full_path

    return None


def compile_pdf(tex_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Compile LaTeX to PDF using available compiler.

    Args:
        tex_path: Path to the .tex file
        output_dir: Output directory (defaults to same as tex file)

    Returns:
        Path to the generated PDF, or None if compilation failed
    """
    tex_path = Path(tex_path).resolve()  # Get absolute path
    output_dir = Path(output_dir).resolve() if output_dir else tex_path.parent

    # Check for LaTeX compiler
    compiler = find_latex_compiler()
    if not compiler:
        print("  Warning: No LaTeX compiler found (pdflatex, xelatex, or lualatex).")
        print("  To generate PDFs, install LaTeX:")
        print("    - Windows: https://miktex.org/download (MiKTeX)")
        print("    - Mac: brew install --cask mactex")
        print("    - Linux: sudo apt install texlive-latex-recommended")
        print(f"  Your .tex file is saved at: {tex_path}")
        return None

    try:
        # Run compiler twice for proper references
        # Use just the filename since we set cwd to the tex file's directory
        tex_filename = tex_path.name
        working_dir = tex_path.parent

        for i in range(2):
            result = subprocess.run(
                [compiler, '-interaction=nonstopmode', tex_filename],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(working_dir)
            )

        pdf_path = working_dir / tex_path.with_suffix('.pdf').name
        if pdf_path.exists():
            # Clean up auxiliary files
            for ext in ['.aux', '.log', '.out', '.toc', '.synctex.gz', '.fls', '.fdb_latexmk']:
                aux_file = working_dir / tex_path.with_suffix(ext).name
                if aux_file.exists():
                    try:
                        aux_file.unlink()
                    except Exception:
                        pass
            return str(pdf_path)
        else:
            # Check log for errors
            log_path = working_dir / tex_path.with_suffix('.log').name
            if log_path.exists():
                log_content = log_path.read_text(encoding='utf-8', errors='ignore')
                # Find first error
                for line in log_content.split('\n'):
                    if line.startswith('!'):
                        print(f"  LaTeX error: {line}")
                        break
            print(f"  Warning: PDF compilation failed. Check {tex_path.stem}.log for details.")
            return None

    except subprocess.TimeoutExpired:
        print("  Warning: PDF compilation timed out (>120s).")
        return None
    except Exception as e:
        print(f"  Warning: PDF compilation error: {e}")
        return None


def render_outreach_md(targets: list, scripts: dict, company: str, role: str) -> str:
    """
    Format outreach plan as markdown with copy-paste scripts.

    Args:
        targets: List of outreach target dicts
        scripts: Dict mapping names to their scripts
        company: Company name
        role: Role title

    Returns:
        Formatted markdown string
    """
    lines = [
        f"# Outreach Plan: {role} @ {company}",
        "",
    ]

    for i, target in enumerate(targets, 1):
        name = target.get('name', 'Unknown')
        lines.extend([
            f"## Contact {i}: {name} ({target.get('connection_type', 'Unknown')} - Priority {target.get('priority', i)})",
            f"**LinkedIn**: {target.get('linkedin_url', 'N/A')}",
            f"**Title**: {target.get('title', 'N/A')}",
            f"**Connection Angle**: {target.get('connection_angle', 'N/A')}",
            "",
        ])

        if name in scripts:
            script = scripts[name]
            if 'connection_request' in script:
                lines.extend([
                    "### Connection Request (copy/paste):",
                    f"> {script['connection_request']}",
                    "",
                ])
            if 'follow_up_message' in script:
                lines.extend([
                    "### Follow-up Message (after accepted):",
                    f"> {script['follow_up_message']}",
                    "",
                ])

        lines.append("---")
        lines.append("")

    # Add outreach sequence if present
    if 'outreach_sequence' in scripts:
        lines.extend([
            "## Recommended Outreach Sequence",
        ])
        for step in scripts['outreach_sequence']:
            lines.append(f"- {step}")
        lines.append("")

    return '\n'.join(lines)


def save_model_inputs(
    job_id: str,
    prompt: str,
    response: dict,
    output_dir: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost: float = 0.0,
    model: str = "claude-sonnet-4-20250514"
) -> None:
    """
    Save model inputs alongside outputs for debugging/auditing.

    Creates:
    - {output_dir}/debug/prompt.md      - Full prompt sent to model
    - {output_dir}/debug/response.json  - Raw model response
    - {output_dir}/debug/metadata.json  - Timestamps, token counts, costs

    Args:
        job_id: Job identifier
        prompt: Full prompt sent to model
        response: Parsed response dict
        output_dir: Base output directory for this job
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        cost: Estimated cost in USD
        model: Model identifier
    """
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Save prompt
    (debug_dir / "prompt.md").write_text(prompt, encoding='utf-8')

    # Save response
    (debug_dir / "response.json").write_text(
        json.dumps(response, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    # Save metadata
    metadata = {
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 4),
    }
    (debug_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding='utf-8'
    )


def generate_daily_summary(results: list, output_dir: str) -> str:
    """
    Create daily digest of all processed jobs.

    Args:
        results: List of ProcessedOutput dicts
        output_dir: Base output directory

    Returns:
        Path to the generated summary file
    """
    today = date.today().isoformat()
    lines = [
        f"# jobflow-ai Results - {today}",
        "",
        f"## Top {len(results)} Ranked Opportunities",
        "",
    ]

    for i, result in enumerate(results, 1):
        analysis = result.get('analysis', {})
        company = analysis.get('company', 'Unknown')
        role = analysis.get('role', 'Unknown')
        job_url = result.get('job_url', '#')

        lines.extend([
            f"### #{i}: {role} - {company}",
            f"**Link**: {job_url}",
            "",
        ])

        # Key requirements
        if analysis.get('key_requirements'):
            lines.append("**Key Requirements**: " + ", ".join(analysis['key_requirements'][:3]))

        # Candidate strengths
        if analysis.get('candidate_strengths'):
            lines.append("**Your Strengths**: " + ", ".join(analysis['candidate_strengths'][:3]))

        lines.append("")

        # Files generated
        job_slug = result.get('slug', slugify(f"{company}-{role}"))
        lines.extend([
            f"**Files**: `{job_slug}/resume.tex`, `{job_slug}/outreach.md`",
            "",
        ])

        # Top contacts
        targets = result.get('outreach_targets', [])
        if targets:
            lines.append("**Top Contacts**:")
            lines.append("| Priority | Name | Type | LinkedIn |")
            lines.append("|----------|------|------|----------|")
            for target in targets[:3]:
                lines.append(
                    f"| {target.get('priority', '-')} | {target.get('name', 'N/A')} | "
                    f"{target.get('connection_type', 'N/A')} | "
                    f"[Profile]({target.get('linkedin_url', '#')}) |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    # Save summary
    summary_path = Path(output_dir) / today / "daily_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text('\n'.join(lines), encoding='utf-8')

    return str(summary_path)


def load_yaml(path: str) -> dict:
    """Load a YAML file."""
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_json(data: dict, path: str) -> None:
    """Save data as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def load_json(path: str) -> dict:
    """Load a JSON file."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


# =============================================================================
# LaTeX Editing Helpers (for variant-based resume tailoring)
# =============================================================================

def replace_summary_in_tex(tex_content: str, new_summary: str) -> str:
    """
    Replace the summary/tagline line in a LaTeX resume.

    Looks for patterns like:
    - \\begin{onecolentry}...summary text...\\end{onecolentry}
    - Single-line summaries after header

    Args:
        tex_content: Original LaTeX content
        new_summary: New summary text (plain text, will be escaped)

    Returns:
        Modified LaTeX content with new summary
    """
    if not new_summary:
        return tex_content

    # Pattern 1: onecolentry with centering (most common in the variants)
    pattern1 = r'(\\begin\{onecolentry\}\s*\\centering\s*).+?(\s*\\end\{onecolentry\})'

    # Check if pattern exists
    if re.search(pattern1, tex_content, re.DOTALL):
        return re.sub(
            pattern1,
            rf'\1{new_summary}\2',
            tex_content,
            count=1,
            flags=re.DOTALL
        )

    # Pattern 2: onecolentry without centering but after header
    pattern2 = r'(\\end\{header\}.*?\\begin\{onecolentry\}\s*).+?(\s*\\end\{onecolentry\})'

    if re.search(pattern2, tex_content, re.DOTALL):
        return re.sub(
            pattern2,
            rf'\1{new_summary}\2',
            tex_content,
            count=1,
            flags=re.DOTALL
        )

    # Pattern 3: Summary in header (AI_safety format)
    pattern3 = r'(\\begin\{header\}.*?\\begin\{onecolentry\}\s*\\centering\s*).+?(\s*\\end\{onecolentry\}.*?\\end\{header\})'

    if re.search(pattern3, tex_content, re.DOTALL):
        return re.sub(
            pattern3,
            rf'\1{new_summary}\2',
            tex_content,
            count=1,
            flags=re.DOTALL
        )

    # If no pattern found, return unchanged
    return tex_content


def reorder_skills_in_tex(tex_content: str, skill_categories_order: list[str]) -> str:
    """
    Reorder skill categories in a LaTeX resume.

    The variants use patterns like:
    \\textbf{Category Name:}
    skill1, skill2, skill3

    Args:
        tex_content: Original LaTeX content
        skill_categories_order: Ordered list of category names to prioritize

    Returns:
        Modified LaTeX content with reordered skills
    """
    if not skill_categories_order:
        return tex_content

    # Find the skills section
    skills_section_match = re.search(
        r'(\\section\{(?:Technical )?Skills\})(.*?)(?=\\section\{|\\end\{document\}|$)',
        tex_content,
        re.DOTALL | re.IGNORECASE
    )

    if not skills_section_match:
        return tex_content

    section_header = skills_section_match.group(1)
    section_content = skills_section_match.group(2)

    # Extract individual skill blocks
    # Pattern: \textbf{Category:} followed by content until next \textbf{ or section end
    skill_pattern = r'(\\textbf\{([^}]+?)(?::|)\}\s*)(.*?)(?=\\textbf\{|\\section\{|$)'
    skill_blocks = re.findall(skill_pattern, section_content, re.DOTALL)

    if not skill_blocks:
        return tex_content

    # Build a dict of category -> (header, content)
    skills_dict = {}
    for header, category_name, content in skill_blocks:
        # Normalize category name for matching
        normalized = category_name.strip().lower().replace('&', 'and').replace(':', '')
        skills_dict[normalized] = (header, content)

    # Reorder based on priority list
    new_blocks = []
    used_categories = set()

    # First, add categories in the specified order
    for category in skill_categories_order:
        normalized = category.strip().lower().replace('&', 'and').replace('_', ' ')
        # Try to find a match
        for key in skills_dict:
            if normalized in key or key in normalized:
                if key not in used_categories:
                    header, content = skills_dict[key]
                    new_blocks.append(f"{header}{content}")
                    used_categories.add(key)
                    break

    # Then add remaining categories in original order
    for key, (header, content) in skills_dict.items():
        if key not in used_categories:
            new_blocks.append(f"{header}{content}")

    # Reconstruct section
    new_section_content = '\n\n'.join(new_blocks)
    new_section = f"{section_header}\n\n{new_section_content}"

    # Replace in original content
    result = tex_content[:skills_section_match.start()] + new_section
    remaining = tex_content[skills_section_match.end():]
    if remaining and not remaining.startswith('\n'):
        result += '\n'
    result += remaining

    return result


def add_skills_to_tex(tex_content: str, skills_to_add: list[str], category: str = "Additional") -> str:
    """
    Add new skills to the skills section.

    Args:
        tex_content: Original LaTeX content
        skills_to_add: List of skills to add
        category: Category name for the new skills

    Returns:
        Modified LaTeX content with added skills
    """
    if not skills_to_add:
        return tex_content

    # Find end of skills section
    skills_section_match = re.search(
        r'(\\section\{(?:Technical )?Skills\}.*?)(\\section\{|\\end\{document\}|$)',
        tex_content,
        re.DOTALL | re.IGNORECASE
    )

    if not skills_section_match:
        return tex_content

    section_content = skills_section_match.group(1)
    next_section = skills_section_match.group(2)

    # Add new skills block
    skills_text = ', '.join(skills_to_add)
    new_block = f"\n\n\\textbf{{{category}:}}\n{skills_text}\n"

    # Insert before next section
    new_content = section_content.rstrip() + new_block + '\n' + next_section

    return tex_content[:skills_section_match.start()] + new_content + tex_content[skills_section_match.end():]


def extract_section_from_tex(tex_content: str, section_name: str) -> Optional[str]:
    """
    Extract a section from LaTeX content.

    Args:
        tex_content: Full LaTeX content
        section_name: Name of section to extract (e.g., "Experience", "Skills")

    Returns:
        Section content or None if not found
    """
    pattern = rf'\\section\{{{section_name}\}}(.*?)(?=\\section\{{|\\end\{{document\}}|$)'
    match = re.search(pattern, tex_content, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def replace_section_in_tex(tex_content: str, section_name: str, new_content: str) -> str:
    """
    Replace a section in LaTeX content.

    Args:
        tex_content: Full LaTeX content
        section_name: Name of section to replace
        new_content: New section content (without \\section header)

    Returns:
        Modified LaTeX content
    """
    pattern = rf'(\\section\{{{section_name}\}})(.*?)((?=\\section\{{)|(?=\\end\{{document\}})|$)'

    def replacer(match):
        header = match.group(1)
        next_part = match.group(3) if match.group(3) else ''
        return f"{header}\n{new_content}\n{next_part}"

    return re.sub(pattern, replacer, tex_content, flags=re.DOTALL | re.IGNORECASE)


def get_output_dir(company: str, role: str, base_dir: str = "outputs") -> Path:
    """
    Get the output directory for a job.

    Args:
        company: Company name
        role: Role title
        base_dir: Base output directory

    Returns:
        Path to the job-specific output directory
    """
    today = date.today().isoformat()
    slug = slugify(f"{company}-{role}")
    path = Path(base_dir) / today / slug
    path.mkdir(parents=True, exist_ok=True)
    return path
