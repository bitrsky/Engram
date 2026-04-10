"""
extract.py — Fact extraction for Engram.

Two modes:
1. Heuristic: Regex-based pattern matching (always available, ~1-2 facts per text)
2. LLM: AI-powered extraction (optional, ~5-7 facts per text)

The LLM mode is the single most impactful AI integration point in the system.
It uses urllib.request directly — no SDK dependencies.
"""

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict

from .config import EngramConfig
from .conflicts import Fact


@dataclass
class FactCandidate:
    """A candidate fact extracted from text."""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.6
    temporal: str = ""  # ISO date if time info found
    source_text: str = ""  # The original sentence/phrase
    conflicts_with: str = ""  # If LLM detected a conflict with known facts


def extract_facts(
    content: str,
    project: str = None,
    existing_facts: List[Fact] = None,
    config: EngramConfig = None,
) -> List[FactCandidate]:
    """
    Extract structured facts from text content.

    Routes to LLM or heuristic based on config.

    Args:
        content: Text to extract facts from
        project: Project context (used in LLM prompt)
        existing_facts: Known facts (fed to LLM for conflict detection)
        config: Configuration (for LLM settings)

    Returns:
        List of FactCandidate objects
    """
    config = config or EngramConfig()
    if config.llm_available:
        try:
            return extract_facts_llm(content, project, existing_facts, config)
        except Exception:
            # LLM failed, fall back to heuristic
            return extract_facts_heuristic(content, project)
    return extract_facts_heuristic(content, project)


def extract_facts_heuristic(content: str, project: str = None) -> List[FactCandidate]:
    """
    Extract facts using regex pattern matching.

    Patterns to match:

    1. Technology choices:
       "use(s|d|ing) X" → (project, uses, X)
       "chose/picked/selected X" → (project, chose, X)
       "built with X" → (project, built_with, X)

    2. Transitions:
       "switch(ed) from X to Y" → (project, uses, Y) + note superseded X
       "migrat(e|ed|ing) from X to Y" → same
       "replac(ed|ing) X with Y" → same

    3. Assignments:
       "X will handle/do/work on Y" → (X, assigned_to, Y)
       "X is responsible for Y" → (X, responsible_for, Y)
       "assigned X to Y" → (X, assigned_to, Y)

    4. Decisions:
       "decided to X" → (project, decision, X)
       "went with X" → (project, chose, X)

    5. Status:
       "X is deployed/live/running" → (X, status, deployed)
       "X is broken/down/failing" → (X, status, broken)

    6. Metrics/numbers:
       "DAU/MAU/users: N" or "about N users" → (project, metric_name, N)
       "costs $X/mo" → (project, cost, $X/mo)

    All heuristic facts get confidence=0.6.
    """
    facts: List[FactCandidate] = []
    proj = project or "project"

    # Normalize content for matching
    sentences = _split_sentences(content)

    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue

        # Pattern 1: Technology choices
        # "uses X", "using X", "used X"
        m = re.search(
            r"\bus(?:es?|ed|ing)\s+(?P<tech>[A-Z][A-Za-z0-9_.+-]+(?:\s+[A-Z][A-Za-z0-9_.+-]+)?)",
            stripped,
        )
        if m:
            tech = _extract_entity(m.group("tech"))
            if tech and len(tech) >= 2:
                facts.append(FactCandidate(
                    subject=proj,
                    predicate="uses",
                    object=tech,
                    confidence=0.6,
                    source_text=stripped,
                ))

        # "chose/picked/selected X"
        m = re.search(
            r"\b(?:chose|picked|selected)\s+(?P<tech>[A-Z][A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            tech = _extract_entity(m.group("tech"))
            if tech and len(tech) >= 2:
                facts.append(FactCandidate(
                    subject=proj,
                    predicate="chose",
                    object=tech,
                    confidence=0.6,
                    source_text=stripped,
                ))

        # "built with X"
        m = re.search(
            r"\bbuilt\s+with\s+(?P<tech>[A-Z][A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            tech = _extract_entity(m.group("tech"))
            if tech and len(tech) >= 2:
                facts.append(FactCandidate(
                    subject=proj,
                    predicate="built_with",
                    object=tech,
                    confidence=0.6,
                    source_text=stripped,
                ))

        # Pattern 2: Transitions
        # "switched from X to Y", "switching from X to Y"
        m = re.search(
            r"\bswitch(?:ed|ing)?\s+from\s+(?P<old>[A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)\s+to\s+(?P<new>[A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            old_tech = _extract_entity(m.group("old"))
            new_tech = _extract_entity(m.group("new"))
            if new_tech and len(new_tech) >= 2:
                facts.append(FactCandidate(
                    subject=proj,
                    predicate="uses",
                    object=new_tech,
                    confidence=0.6,
                    source_text=stripped,
                    conflicts_with=f"{proj} → uses → {old_tech}" if old_tech else "",
                ))

        # "migrated from X to Y", "migrating from X to Y"
        m = re.search(
            r"\bmigrat(?:e|ed|ing)\s+from\s+(?P<old>[A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)\s+to\s+(?P<new>[A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            old_tech = _extract_entity(m.group("old"))
            new_tech = _extract_entity(m.group("new"))
            if new_tech and len(new_tech) >= 2:
                facts.append(FactCandidate(
                    subject=proj,
                    predicate="uses",
                    object=new_tech,
                    confidence=0.6,
                    source_text=stripped,
                    conflicts_with=f"{proj} → uses → {old_tech}" if old_tech else "",
                ))

        # "replaced X with Y", "replacing X with Y"
        m = re.search(
            r"\breplac(?:ed|ing)\s+(?P<old>[A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)\s+with\s+(?P<new>[A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            old_tech = _extract_entity(m.group("old"))
            new_tech = _extract_entity(m.group("new"))
            if new_tech and len(new_tech) >= 2:
                facts.append(FactCandidate(
                    subject=proj,
                    predicate="uses",
                    object=new_tech,
                    confidence=0.6,
                    source_text=stripped,
                    conflicts_with=f"{proj} → uses → {old_tech}" if old_tech else "",
                ))

        # Pattern 3: Assignments
        # "X will handle/do/work on Y"
        m = re.search(
            r"\b(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+will\s+(?:handle|do|work\s+on)\s+(?P<task>.+?)(?:\.|$)",
            stripped,
        )
        if m:
            person = _extract_entity(m.group("person"))
            task = m.group("task").strip().rstrip(".")
            if person and task:
                facts.append(FactCandidate(
                    subject=person,
                    predicate="assigned_to",
                    object=task,
                    confidence=0.6,
                    source_text=stripped,
                ))

        # "X is responsible for Y"
        m = re.search(
            r"\b(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+is\s+responsible\s+for\s+(?P<task>.+?)(?:\.|$)",
            stripped,
        )
        if m:
            person = _extract_entity(m.group("person"))
            task = m.group("task").strip().rstrip(".")
            if person and task:
                facts.append(FactCandidate(
                    subject=person,
                    predicate="responsible_for",
                    object=task,
                    confidence=0.6,
                    source_text=stripped,
                ))

        # "assigned X to Y"
        m = re.search(
            r"\bassigned\s+(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+to\s+(?P<task>.+?)(?:\.|$)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            person = _extract_entity(m.group("person"))
            task = m.group("task").strip().rstrip(".")
            if person and task:
                facts.append(FactCandidate(
                    subject=person,
                    predicate="assigned_to",
                    object=task,
                    confidence=0.6,
                    source_text=stripped,
                ))

        # Pattern 4: Decisions
        # "decided to X"
        m = re.search(
            r"\bdecided\s+to\s+(?P<decision>.+?)(?:\.|$)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            decision = m.group("decision").strip().rstrip(".")
            if decision and len(decision) >= 3:
                facts.append(FactCandidate(
                    subject=proj,
                    predicate="decision",
                    object=decision,
                    confidence=0.6,
                    source_text=stripped,
                ))

        # "went with X"
        m = re.search(
            r"\bwent\s+with\s+(?P<choice>[A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            choice = _extract_entity(m.group("choice"))
            if choice and len(choice) >= 2:
                facts.append(FactCandidate(
                    subject=proj,
                    predicate="chose",
                    object=choice,
                    confidence=0.6,
                    source_text=stripped,
                ))

        # Pattern 5: Status
        # "X is deployed/live/running"
        m = re.search(
            r"\b(?P<entity>[A-Z][A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)\s+is\s+(?P<status>deployed|live|running|complete|finished|done|ready|stable)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            entity = _extract_entity(m.group("entity"))
            status = m.group("status").lower()
            if entity and len(entity) >= 2:
                facts.append(FactCandidate(
                    subject=entity,
                    predicate="status",
                    object=status,
                    confidence=0.6,
                    source_text=stripped,
                ))

        # "X is broken/down/failing"
        m = re.search(
            r"\b(?P<entity>[A-Z][A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)\s+is\s+(?P<status>broken|down|failing|crashed|offline|blocked|stalled|stuck)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            entity = _extract_entity(m.group("entity"))
            status = m.group("status").lower()
            if entity and len(entity) >= 2:
                facts.append(FactCandidate(
                    subject=entity,
                    predicate="status",
                    object=status,
                    confidence=0.6,
                    source_text=stripped,
                ))

        # Pattern 6: Metrics/numbers
        # "DAU: N", "MAU: N", "users: N" or "N users/DAU/MAU"
        m = re.search(
            r"\b(?P<metric>DAU|MAU|users|active\s+users)[:\s]+(?:about\s+)?(?P<value>[\d,]+(?:\.\d+)?(?:\s*[kKmMbB])?)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            metric = m.group("metric").strip().lower().replace(" ", "_")
            value = m.group("value").strip()
            facts.append(FactCandidate(
                subject=proj,
                predicate=metric,
                object=value,
                confidence=0.6,
                source_text=stripped,
            ))

        # "about N users"
        if not m:
            m2 = re.search(
                r"\babout\s+(?P<value>[\d,]+(?:\.\d+)?(?:\s*[kKmMbB])?)\s+(?P<metric>users|DAU|MAU|customers|subscribers)",
                stripped,
                re.IGNORECASE,
            )
            if m2:
                metric = m2.group("metric").strip().lower()
                value = m2.group("value").strip()
                facts.append(FactCandidate(
                    subject=proj,
                    predicate=metric,
                    object=value,
                    confidence=0.6,
                    source_text=stripped,
                ))

        # "costs $X/mo" or "costs $X per month"
        m = re.search(
            r"\bcosts?\s+(?P<value>\$[\d,.]+(?:/mo(?:nth)?|\s*per\s*month)?)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            value = m.group("value").strip()
            facts.append(FactCandidate(
                subject=proj,
                predicate="cost",
                object=value,
                confidence=0.6,
                source_text=stripped,
            ))

    return facts


def extract_facts_llm(
    content: str,
    project: str = None,
    existing_facts: List[Fact] = None,
    config: EngramConfig = None,
) -> List[FactCandidate]:
    """
    Extract facts using LLM.

    The prompt feeds in known facts so the LLM can:
    1. Extract new facts
    2. Detect conflicts with known facts (conflict detection for free)
    3. Infer temporal information

    Uses urllib.request directly — no SDK.

    Cost estimate: ~$0.001 per call with Haiku/GPT-4o-mini
    """
    config = config or EngramConfig()

    # Build the prompt
    prompt = _build_extraction_prompt(content, project, existing_facts)

    # Call LLM
    response_text = _call_llm(prompt, config)
    if not response_text:
        return []

    # Parse JSON response
    return _parse_llm_response(response_text)


def _build_extraction_prompt(
    content: str,
    project: str = None,
    existing_facts: List[Fact] = None,
) -> str:
    """
    Build the fact extraction prompt.

    The prompt structure:
    1. System instruction (extract facts as triples)
    2. Known facts context (if available)
    3. The content to analyze
    4. Output format specification (JSON array)
    """
    known_facts_section = ""
    if existing_facts:
        facts_lines = []
        for f in existing_facts[:20]:  # Limit to 20 to save tokens
            facts_lines.append(f"  - {f.subject} → {f.predicate} → {f.object}")
        known_facts_section = f"""
Known facts about this project:
{chr(10).join(facts_lines)}

If any extracted fact contradicts a known fact above, note it in "conflicts_with".
"""

    return f"""Extract structured facts from the following text.

Project context: {project or "(unknown)"}
{known_facts_section}
Text to analyze:
---
{content}
---

Extract ALL factual claims as (subject, predicate, object) triples.
Include: decisions, assignments, technical choices, timelines, concerns, metrics, relationships.
Skip: opinions about code quality, generic statements, pleasantries, hypotheticals.

For each fact:
- subject: the entity (person, project, component)
- predicate: the relationship (uses, assigned_to, decided, status, etc.)
- object: the value
- confidence: 0.0-1.0 based on how certain this fact is from the text
- temporal: ISO date or relative time phrase if present, empty string otherwise
- conflicts_with: "subject → predicate → object" of a conflicting known fact, or empty string

Output ONLY a valid JSON array, no markdown formatting:
[
  {{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9, "temporal": "", "conflicts_with": ""}}
]"""


def _call_llm(prompt: str, config: EngramConfig) -> Optional[str]:
    """
    Call LLM via HTTP API. Supports ollama, openai, anthropic.
    Uses urllib.request — no SDK dependencies.

    Returns: response text, or None on failure
    """
    provider = config.llm_provider

    if provider == "none":
        return None

    if provider == "ollama":
        return _call_ollama(prompt, config)
    elif provider == "openai":
        return _call_openai(prompt, config)
    elif provider == "anthropic":
        return _call_anthropic(prompt, config)

    return None


def _call_ollama(prompt: str, config: EngramConfig) -> Optional[str]:
    """Call Ollama API at localhost:11434."""
    base_url = config.llm_base_url or "http://localhost:11434"
    url = f"{base_url}/api/generate"
    payload = {
        "model": config.llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},  # Low temp for structured extraction
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError,
            ValueError, KeyError):
        return None


def _call_openai(prompt: str, config: EngramConfig) -> Optional[str]:
    """Call OpenAI-compatible API."""
    base_url = config.llm_base_url or "https://api.openai.com"
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": config.llm_model,
        "messages": [
            {"role": "system", "content": "You are a fact extraction engine. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.llm_api_key}",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError,
            ValueError, KeyError):
        return None


def _call_anthropic(prompt: str, config: EngramConfig) -> Optional[str]:
    """Call Anthropic API."""
    base_url = config.llm_base_url or "https://api.anthropic.com"
    url = f"{base_url}/v1/messages"
    payload = {
        "model": config.llm_model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": config.llm_api_key,
        "anthropic-version": "2023-06-01",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content_blocks = body.get("content", [])
            if content_blocks:
                # Anthropic returns an array of content blocks; grab first text block
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
            return None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError,
            ValueError, KeyError):
        return None


def _parse_llm_response(response_text: str) -> List[FactCandidate]:
    """
    Parse LLM JSON response into FactCandidate objects.

    Handles:
    - Clean JSON array
    - JSON wrapped in ```json ... ``` markdown
    - Partial/malformed JSON (best effort)

    All LLM facts get confidence from the LLM output (or 0.9 default).
    """
    text = response_text.strip()

    # Strip markdown code block wrappers: ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # Try to find the JSON array in the text (in case there is preamble/postamble)
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
        text = text[bracket_start : bracket_end + 1]

    # Fix common JSON issues: trailing commas before ] or }
    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Last resort: try to fix more aggressively
        # Remove any non-JSON text lines before/after
        lines = text.splitlines()
        json_lines = []
        inside = False
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("[") or stripped_line.startswith("{"):
                inside = True
            if inside:
                json_lines.append(line)
            if stripped_line.endswith("]") and inside:
                break
        if json_lines:
            cleaned = "\n".join(json_lines)
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(parsed, list):
        return []

    facts: List[FactCandidate] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue

        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()

        # Skip entries missing required fields
        if not subject or not predicate or not obj:
            continue

        confidence = 0.9  # default for LLM-extracted facts
        raw_confidence = item.get("confidence")
        if raw_confidence is not None:
            try:
                confidence = float(raw_confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.9

        temporal = str(item.get("temporal", "") or "").strip()
        conflicts_with = str(item.get("conflicts_with", "") or "").strip()

        facts.append(FactCandidate(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            temporal=temporal,
            source_text="",  # LLM doesn't echo back the source sentence
            conflicts_with=conflicts_with,
        ))

    return facts


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences for pattern matching. Handle common edge cases."""
    # First, split on newlines to separate paragraphs/lines
    lines = text.split("\n")
    sentences: List[str] = []

    # Common abbreviations that should NOT trigger a split
    # We protect them by temporarily replacing the dots
    _ABBREVS = [
        "e.g.", "i.e.", "vs.", "etc.", "Dr.", "Mr.", "Mrs.", "Ms.",
        "Jr.", "Sr.", "Inc.", "Ltd.", "Corp.", "Prof.", "Gen.",
        "approx.", "dept.", "est.", "fig.", "govt.", "misc.",
        "no.", "pt.", "vol.",
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Protect abbreviations by replacing dots with a placeholder
        protected = line
        placeholders: Dict[str, str] = {}
        for i, abbr in enumerate(_ABBREVS):
            placeholder = f"\x00ABBR{i}\x00"
            if abbr.lower() in protected.lower():
                # Case-insensitive replacement while preserving original case
                pattern = re.compile(re.escape(abbr), re.IGNORECASE)
                matches = pattern.findall(protected)
                for match in matches:
                    placeholders[placeholder] = match
                    protected = protected.replace(match, placeholder, 1)

        # Protect decimal numbers (e.g., "3.14", "100.5")
        decimal_placeholders: Dict[str, str] = {}
        decimal_pattern = re.compile(r"(\d+\.\d+)")
        for j, dm in enumerate(decimal_pattern.findall(protected)):
            ph = f"\x00DEC{j}\x00"
            decimal_placeholders[ph] = dm
            protected = protected.replace(dm, ph, 1)

        # Now split on sentence-ending punctuation: . ! ?
        # Only split on . when followed by whitespace and an uppercase letter, or end of string
        parts = re.split(r"(?<=[.!?])\s+", protected)

        for part in parts:
            # Restore placeholders
            restored = part
            for ph, original in placeholders.items():
                restored = restored.replace(ph, original)
            for ph, original in decimal_placeholders.items():
                restored = restored.replace(ph, original)
            restored = restored.strip()
            if restored:
                sentences.append(restored)

    return sentences


def _extract_entity(text: str) -> str:
    """
    Extract a likely entity name from a text fragment.
    Handles: "Clerk", "MongoDB", "the frontend", "Maya", "our API"
    Returns the cleaned entity name.
    """
    cleaned = text.strip()

    # Remove surrounding quotes (single, double, backticks)
    cleaned = cleaned.strip("\"'`")

    # Remove leading articles
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)

    # Remove leading possessives
    cleaned = re.sub(r"^(?:our|my|their|his|her|its|your)\s+", "", cleaned, flags=re.IGNORECASE)

    # Remove trailing punctuation
    cleaned = cleaned.rstrip(".,;:!?")

    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
