"""
quality.py — Quality gate for Engram memories.

Filters out noise, boilerplate, and low-value content before storage.
No LLM required — pure heuristic rules.

Patterns are configurable via ``~/.engram/patterns.toml``.  Built-in English
patterns are used by default; users can extend them with additional patterns
(e.g. Chinese, Japanese) or replace them entirely by setting ``replace = true``.
"""

import re
from typing import TYPE_CHECKING, Optional, Set, Tuple

if TYPE_CHECKING:
    from .config import EngramConfig


def quality_gate(
    content: str,
    source_type: str = "note",
    config: Optional["EngramConfig"] = None,
) -> Tuple[bool, float]:
    """
    Evaluate whether content is worth storing and assign importance.

    Args:
        content: The text to evaluate
        source_type: "note" | "conversation" | "code" | "document"
        config: EngramConfig instance (optional — uses defaults if None)

    Returns:
        (should_store: bool, importance: float)
        importance range: 0.5 (low) to 5.0 (high), default 3.0
    """
    should_store, importance, _ = quality_gate_detailed(content, source_type, config)
    return (should_store, importance)


def quality_gate_detailed(
    content: str,
    source_type: str = "note",
    config: Optional["EngramConfig"] = None,
) -> Tuple[bool, float, Set[str]]:
    """
    Like :func:`quality_gate` but also returns which pattern categories matched.

    Returns:
        (should_store, importance, matched_categories)
        matched_categories is a set like {"decision_markers", "milestone_markers"}
    """
    # ── Rule 1: Empty or whitespace-only ──
    if not content or not content.strip():
        return (False, 0.0, set())

    stripped = content.strip()
    lower = stripped.lower()

    # ── Load patterns from config (or builtins) ──
    if config is not None:
        decision_markers = config.quality_decision_markers
        milestone_markers = config.quality_milestone_markers
        problem_markers = config.quality_problem_markers
        noise_patterns = config.quality_noise_patterns
    else:
        from .config import _BUILTIN_PATTERNS

        _qp = _BUILTIN_PATTERNS["quality"]
        decision_markers = _qp["decision_markers"]
        milestone_markers = _qp["milestone_markers"]
        problem_markers = _qp["problem_markers"]
        noise_patterns = _qp["noise_patterns"]

    # ── Rule 4: Noise patterns ──
    for pattern in noise_patterns:
        if re.search(pattern, lower, re.IGNORECASE):
            return (False, 0.0, {"noise_patterns"})

    # ── Rule 2: Code files with low prose ratio ──
    if source_type == "code":
        prose_ratio = _count_prose_ratio(stripped)
        if prose_ratio < 0.15:
            return (False, 0.0, set())

    # ── Check for memory markers ──
    matched: Set[str] = set()
    has_decision = any(re.search(p, lower) for p in decision_markers)
    has_milestone = any(re.search(p, lower) for p in milestone_markers)
    has_problem = any(re.search(p, lower) for p in problem_markers)

    if has_decision:
        matched.add("decision_markers")
    if has_milestone:
        matched.add("milestone_markers")
    if has_problem:
        matched.add("problem_markers")

    has_any_marker = bool(matched)

    # ── Rule 3: Too short (< 50 chars) with no markers ──
    if len(stripped) < 50 and not has_any_marker:
        return (True, 0.5, matched)

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

    return (True, importance, matched)


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
