"""Rule-based section tagging of posting text: which part of the posting a mention sits in.

A heading is a short standalone line (not a bullet) whose text matches one of the section patterns
below. Lines that look like headings but match nothing (e.g. a company tagline) are not headings.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

# Order matters: first match wins. Responsibilities precede requirements so that
# "What you'll do (role expectations)" is not read as requirements.
_SECTION_PATTERNS: list[tuple[str, str]] = [
    ("legal", r"equal (employment )?opportunit|diversity|inclusion|inclusive|accommodation|"
              r"privacy|pay transparency|compliance|\beeo\b|disclaimer|fraud|recruitment scam"),
    ("benefits", r"benefit|perks|what we offer|why join|why work|total rewards|compensation|"
                 r"what you can expect from us|life at|what('|’)s in it for you"),
    ("responsibilities", r"responsibilit|what you('|’)ll do|what you will do|you('|’)ll be doing|"
                         r"you will be doing|about the role|the role|^role\b|"
                         r"(role|job|position) overview|^overview$|day.to.day|the impact|"
                         r"job summary|job description|in this role|the opportunity|"
                         r"your mission|key result"),
    ("requirements", r"requirement|qualification|what you bring|what you('|’)ll bring|"
                     r"looking for|what we look for|who you are|you have|you bring|skills|"
                     r"must.have|nice.to.have|bonus|preferred|^experience|"
                     r"(required|relevant|your|minimum|preferred) experience|education|"
                     r"ideal candidate|what you('|’)ll need|what you will need|expectations|"
                     r"success profile|about you"),
    ("about", r"^about|who we are|our mission|our story|our culture|the .+ experience|"
              r"^the team$|company"),
]  # fmt: skip
_COMPILED = [(name, re.compile(p, re.I)) for name, p in _SECTION_PATTERNS]
MAX_HEADING_CHARS = 70
MAX_HEADING_WORDS = 10


def classify_heading(line: str) -> str | None:
    s = line.strip()
    if not s or len(s) > MAX_HEADING_CHARS or s.startswith(("-", "•", "○", "*")):
        return None
    if len(s.split()) > MAX_HEADING_WORDS or (s.endswith(".") and not s.endswith("...")):
        return None
    text = s.rstrip(":").strip()
    for name, rx in _COMPILED:
        if rx.search(text):
            return name
    return None


@dataclass(frozen=True)
class Sections:
    starts: list[int]  # char offsets where each section begins (sorted)
    labels: list[str]

    def at(self, char_offset: int) -> str:
        i = bisect.bisect_right(self.starts, char_offset) - 1
        return self.labels[i]

    @property
    def n_headings(self) -> int:
        return len(self.starts) - 1


def tag_sections(text: str, first_label: str = "intro") -> Sections:
    starts, labels = [0], [first_label]
    offset = 0
    for line in text.splitlines(keepends=True):
        label = classify_heading(line)
        if label is not None:
            starts.append(offset)
            labels.append(label)
        offset += len(line)
    return Sections(starts, labels)
