"""The README's "Code layout" listed none of the ten modules added since October
(red_flags, contraindications, interactions, comparison, brands, symptom_plan,
age_hints, spellfix, hinglish, safety_topics) — found Nov 17. This keeps the list
and the code in step: every module under app/ is named, and every file the README
names exists."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

_NOT_LISTED = {"__init__"}


def test_every_app_module_is_named_in_the_readme():
    missing = [
        path.name for path in sorted((ROOT / "app").glob("*.py"))
        if path.stem not in _NOT_LISTED and f"app/{path.name}" not in README
    ]
    assert not missing, f"not in the README's code layout: {missing}"


def test_every_file_the_readme_names_exists():
    named = set(re.findall(r"`((?:app|tests|static)/[A-Za-z0-9_./-]+\.(?:py|js|html|css|yml))`", README))
    assert named, "the README names no files at all?"
    missing = sorted(p for p in named if not (ROOT / p).exists())
    assert not missing, f"the README names files that do not exist: {missing}"
