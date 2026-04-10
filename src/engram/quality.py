"""
quality.py — Quality gate for Engram memories.

Filters out noise, boilerplate, and low-value content before storage.
No LLM required — pure heuristic rules.
"""

import re
from typing import Tuple


# Memory markers for importance boosting (reused from mempalace's general_extractor patterns)
DECISION_MARKERS = [
    r"\b(decided|chose|went with|picked|settled on|switched to)\b",
    r"\b(instead of|rather than|over.*because)\b",
    r"\b(trade-?off|pros and cons)\b",
]

MILESTONE_MARKERS = [
    r"\b(shipped|launched|deployed|released|it works|got it working)\b",
    r"\b(breakthrough|figured out|nailed it|cracked it)\b",
    r"\b(finally|first time|first ever)\b",
]

PROBLEM_MARKERS = [
    r"\b(the fix was|root cause|the problem was|solved by)\b",
    r"\b(workaround|the answer was|resolved)\b",
]

NOISE_PATTERNS = [
    r"^(ok|okay|sure|got it|thanks|thank you|yes|no|right|fine|cool|nice|great|yep|nope|alright)\s*[.!?]?\s*$",
    r"^(sounds good|makes sense|understood|will do|on it|roger|ack|noted)\s*[.!?]?\s*$",
    r"^(here'?s|let me|sure,?\s*i'?ll|i'?ll help|certainly|of course|absolutely)[,!.\s]",
    r"^(i'?d be happy to|i can help|great question|good point)\s",
]


def quality_gate(content: str, source_type: str = "note") -> Tuple[bool, float]:
    """
    Evaluate whether content is worth storing and assign importance.

    Args:
        content: The text to evaluate
        source_type: "note" | "conversation" | "code" | "document"

    Returns:
        (should_store: bool, importance: float)
        importance range: 0.5 (low) to 5.0 (high), default 3.0

    Rules:
    1. Empty or whitespace-only → (False, 0)
    2. Code files with prose_ratio < 0.15 → (False, 0)
       prose_ratio = count of alphabetic chars in prose lines / total chars
    3. Too short (< 50 chars) AND no memory markers → (True, 0.5)
    4. Noise patterns → (False, 0):
       - Single-word responses: "ok", "sure", "got it", "thanks", "yes", "no", "right"
       - AI template openings: "Here's", "Let me", "Sure, I'll", "I'll help", "Certainly!"
       - Pure acknowledgments: "Sounds good", "Makes sense", "Understood"
    5. Has strong memory markers → importance boost:
       - Decision markers ("decided", "chose", "went with") → +1.0
       - Milestone markers ("shipped", "launched", "finally works") → +1.0
       - Problem+solution markers ("the fix was", "root cause") → +0.5
    6. Default → (True, 3.0)
    """
    # ── Rule 1: Empty or whitespace-only ──
    if not content or not content.strip():
        return (False, 0.0)

    stripped = content.strip()
    lower = stripped.lower()

    # ── Rule 4: Noise patterns (check before short-content rule so
    #    "ok" doesn't sneak through as a short-but-stored note) ──
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return (False, 0.0)

    # ── Rule 2: Code files with low prose ratio ──
    if source_type == "code":
        prose_ratio = _count_prose_ratio(stripped)
        if prose_ratio < 0.15:
            return (False, 0.0)

    # ── Helper: check for any memory markers ──
    has_decision = any(re.search(p, lower) for p in DECISION_MARKERS)
    has_milestone = any(re.search(p, lower) for p in MILESTONE_MARKERS)
    has_problem = any(re.search(p, lower) for p in PROBLEM_MARKERS)
    has_any_marker = has_decision or has_milestone or has_problem

    # ── Rule 3: Too short (< 50 chars) with no markers ──
    if len(stripped) < 50 and not has_any_marker:
        return (True, 0.5)

    # ── Rule 5 & 6: Compute importance with marker boosts ──
    importance = 3.0

    if has_decision:
        importance += 1.0
    if has_milestone:
        importance += 1.0
    if has_problem:
        importance += 0.5

    # Cap at 5.0
    importance = min(importance, 5.0)

    return (True, importance)


def _count_prose_ratio(content: str) -> float:
    """
    Calculate the ratio of prose (natural language) vs code in the content.

    Skips lines that look like code:
    - Lines starting with import/from/def/class/function/const/let/var/return
    - Lines starting with $ # (shell prompts)
    - Lines inside ``` blocks
    - Lines that are mostly non-alphabetic (operators, brackets)

    Returns: float 0.0-1.0
    """
    if not content:
        return 0.0

    total_chars = len(content)
    if total_chars == 0:
        return 0.0

    lines = content.splitlines()

    # Pattern for lines that look like code statements
    code_line_re = re.compile(
        r"^\s*"
        r"(import |from |def |class |function |const |let |var |return |"
        r"if |elif |else:|for |while |try:|except |catch |switch |case |"
        r"async |await |yield |raise |throw |export |module\.)"
    )
    # Shell prompt lines
    shell_re = re.compile(r"^\s*[\$#>]\s")

    prose_alpha_count = 0
    in_code_block = False

    for line in lines:
        stripped_line = line.strip()

        # Track fenced code blocks
        if stripped_line.startswith("```"):
            in_code_block = not in_code_block
            continue

        # Skip everything inside ``` blocks
        if in_code_block:
            continue

        # Skip empty lines
        if not stripped_line:
            continue

        # Skip shell prompts
        if shell_re.match(line):
            continue

        # Skip lines matching code patterns
        if code_line_re.match(line):
            continue

        # Skip lines that are mostly non-alphabetic (operators, brackets, etc.)
        # A line is "mostly non-alphabetic" if fewer than 40% of its non-space
        # characters are letters.
        non_space = re.sub(r"\s", "", stripped_line)
        if non_space:
            alpha_in_line = sum(1 for c in non_space if c.isalpha())
            if alpha_in_line / len(non_space) < 0.4:
                continue

        # This line looks like prose — count its alphabetic characters
        prose_alpha_count += sum(1 for c in stripped_line if c.isalpha())

    return prose_alpha_count / total_chars
