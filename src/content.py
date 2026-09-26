"""Reads content/dr_facts.md, the hand-written text shared by the Home page and the chatbot.
Sections are split on '# ' headings, so the file stays the single source of every sentence."""
import re

from src import config as C

FACTS = C.CONTENT_DIR / "dr_facts.md"
_GRADE = re.compile(r"^- \*\*Grade (\d), ([^:]+):\*\*\s*(.*)$")


def sections() -> dict[str, str]:
    text = re.sub(r"<!--.*?-->", "", FACTS.read_text(encoding="utf-8"), flags=re.S)
    out, title, body = {}, None, []
    for line in text.splitlines():
        if line.startswith("# "):
            if title:
                out[title] = "\n".join(body).strip()
            title, body = line[2:].strip(), []
        else:
            body.append(line)
    if title:
        out[title] = "\n".join(body).strip()
    return out


def grade_definitions(section: str) -> dict[int, tuple[str, str]]:
    """{grade: (name, definition)} from the '- **Grade N, Name:** text' bullets, joining
    wrapped continuation lines."""
    out, current = {}, None
    for line in section.splitlines():
        m = _GRADE.match(line)
        if m:
            current = int(m.group(1))
            out[current] = (m.group(2).strip(), m.group(3).strip())
        elif current is not None and line.startswith("  ") and line.strip():
            name, text = out[current]
            out[current] = (name, f"{text} {line.strip()}")
        elif not line.strip():
            current = None
    return out


def split_paragraphs(section: str) -> tuple[str, str]:
    """(text before the first bullet list, the rest) - lets a page place a visual in between."""
    lines = section.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("- "):
            return "\n".join(lines[:i]).strip(), "\n".join(lines[i:]).strip()
    return section, ""


def after_bullets(section: str) -> str:
    """Text that follows the first bullet list in a section (notes, sources)."""
    lines = section.splitlines()
    seen, end = False, None
    for i, line in enumerate(lines):
        if line.startswith("- "):
            seen = True
        elif seen and line.strip() and not line.startswith("  "):
            end = i
            break
    return "\n".join(lines[end:]).strip() if end is not None else ""
