"""Skill loader untuk .agent/skills/*/SKILL.md"""
import pathlib
import re
from typing import Dict, List

from .config import SKILLS_DIR

def _parse_skill_file(path: pathlib.Path) -> Dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"name": path.parent.name, "description": f"error read: {e}", "content": ""}
    # coba parse frontmatter --- yaml ---
    description = ""
    # ambil heading pertama sebagai deskripsi fallback
    # cari frontmatter
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if fm_match:
        fm = fm_match.group(1)
        # cari description: field
        m = re.search(r"description\s*:\s*(.+)", fm, re.IGNORECASE)
        if m:
            description = m.group(1).strip().strip('"').strip("'")
        m2 = re.search(r"name\s*:\s*(.+)", fm, re.IGNORECASE)
        name = m2.group(1).strip().strip('"').strip("'") if m2 else path.parent.name
        content = text[fm_match.end():]
    else:
        name = path.parent.name
        content = text
        # cari # Title
        h1 = re.search(r"^#\s+(.+)", content, re.MULTILINE)
        if h1:
            description = h1.group(1).strip()
        else:
            # first non-empty line
            for line in content.splitlines():
                line=line.strip()
                if line and not line.startswith("#"):
                    description = line[:120]
                    break
    if not description:
        description = content[:120].strip().replace("\n"," ")
    return {"name": name, "description": description, "content": content.strip(), "path": str(path)}

def discover_skills(skills_dir: str = SKILLS_DIR) -> Dict[str, Dict[str, str]]:
    base = pathlib.Path(skills_dir)
    skills: Dict[str, Dict[str, str]] = {}
    if not base.exists():
        return skills
    for child in base.iterdir():
        if not child.is_dir():
            continue
        # cari SKILL.md case-insensitive
        candidates = list(child.glob("SKILL.md")) + list(child.glob("skill.md")) + list(child.glob("SKILL.MD"))
        if not candidates:
            # cari *.md apapun di folder
            candidates = list(child.glob("*.md"))
        if not candidates:
            continue
        info = _parse_skill_file(candidates[0])
        # key = folder name lower
        key = child.name
        skills[key] = info
    return skills

def list_skills(skills_dir: str = SKILLS_DIR) -> str:
    skills = discover_skills(skills_dir)
    if not skills:
        return f"(no skills) direktori {skills_dir} kosong atau tidak ada. Buat folder .agent/skills/<nama>/SKILL.md untuk menambah skill."
    lines = [f"Available skills ({len(skills)}):"]
    for name, info in sorted(skills.items()):
        lines.append(f"- {name}: {info['description']}  [path: {info['path']}]")
    lines.append("\nGunakan skill_load(name) untuk memuat panduan lengkap.")
    return "\n".join(lines)

def load_skill(name: str, skills_dir: str = SKILLS_DIR) -> str:
    skills = discover_skills(skills_dir)
    # case-insensitive
    key = None
    for k in skills:
        if k.lower() == name.lower():
            key = k
            break
    if not key:
        return f"Error: skill '{name}' tidak ditemukan. Available: {', '.join(sorted(skills.keys())) or '(none)'}"
    info = skills[key]
    header = f"# Skill: {info['name']}\nPath: {info['path']}\nDescription: {info['description']}\n\n"
    return header + info["content"]

def build_skills_catalog(skills_dir: str = SKILLS_DIR) -> str:
    skills = discover_skills(skills_dir)
    if not skills:
        return ""
    lines = []
    for name, info in sorted(skills.items()):
        lines.append(f"- **{name}**: {info['description']}")
    return "\n".join(lines)
