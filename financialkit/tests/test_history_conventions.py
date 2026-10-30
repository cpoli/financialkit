"""Documentation contract: every domain's history page lists at least 15
dated breakthroughs in chronological order, each with an *Implementation:*
line and its own gallery example(s), and every gallery example belongs to
exactly one breakthrough.

Skipped outside a source checkout (e.g. an installed wheel without docs/).
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "docs" / "source" / "history"
EXAMPLES = ROOT / "examples"

pytestmark = pytest.mark.skipif(not HISTORY.is_dir() or not EXAMPLES.is_dir(), reason="needs a source checkout")

DOMAINS = sorted(p.name for p in EXAMPLES.iterdir() if p.is_dir()) if EXAMPLES.is_dir() else []
MINIMUM_BREAKTHROUGHS = 15
_HEADING = re.compile(r"^(?P<title>[^\n]+)\n-{3,}\n", re.M)
_DATE = re.compile(r"^(?:c\. )?(?P<year>\d{4}) -- \S")


def _breakthroughs(domain):
    text = (HISTORY / f"{domain}_breakthroughs.rst").read_text()
    matches = list(_HEADING.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        links = [p for line in re.findall(r"^\.\. minigallery::(.*)$", body, re.M) for p in line.split()]
        if match.group("title").strip() != "See Also":
            sections.append((match.group("title").strip(), body, links))
    return sections


def _linked_paths():
    return [(domain, title, (ROOT / "docs" / "source" / link).resolve()) for domain in DOMAINS for title, _, links in _breakthroughs(domain) for link in links]


@pytest.mark.parametrize("domain", DOMAINS)
def test_history_has_enough_dated_breakthroughs_in_order(domain):
    sections = _breakthroughs(domain)
    assert len(sections) >= MINIMUM_BREAKTHROUGHS, f"{domain}: only {len(sections)} breakthroughs"
    years = []
    for title, _, _ in sections:
        match = _DATE.match(title)
        assert match, f"{domain}: '{title}' is not headed 'YYYY -- ...'"
        years.append(int(match.group("year")))
    assert years == sorted(years), f"{domain}: breakthroughs out of chronological order {years}"


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_breakthrough_has_implementation_and_gallery(domain):
    for title, body, links in _breakthroughs(domain):
        assert re.search(r"^\*Implementation:\*", body, re.M), f"{domain}: '{title}' has no *Implementation:* line"
        assert links, f"{domain}: '{title}' has no minigallery example"


def test_every_linked_example_exists():
    missing = [f"{domain}: '{title}' -> {path}" for domain, title, path in _linked_paths() if not path.is_file()]
    assert not missing, f"minigallery targets do not exist: {missing}"


def test_every_example_is_linked_from_exactly_one_breakthrough():
    linked = [path for _, _, path in _linked_paths()]
    twice = sorted({str(p) for p in linked if linked.count(p) > 1})
    assert not twice, f"examples linked from more than one breakthrough: {twice}"
    on_disk = {p.resolve() for p in EXAMPLES.rglob("plot_*.py")}
    unlinked = sorted(map(str, on_disk - set(linked)))
    assert not unlinked, f"examples not linked from any breakthrough: {unlinked}"
