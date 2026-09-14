import json
import re
import time
import requests
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from src.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_MAX_RETRIES, OLLAMA_TIMEOUT_SECONDS

NUM_PREDICT_CAP = 4096

class CareerStage(BaseModel):
    stage_order: int = Field(..., description="1-based order of stage in life/career")
    occupation_clean: str = Field(..., description="normalized occupation label (general)")
    evidence: str = Field(..., description="short quote/paraphrase from biography supporting this stage")
    start_year: Optional[int] = Field(None, description="only if explicitly stated")
    end_year: Optional[int] = Field(None, description="only if explicitly stated")

class CareerTimeline(BaseModel):
    qid: str
    occupations_input: List[str]
    occupations_clean: List[str]
    stages: List[CareerStage]
    notes: str

SYSTEM_RULES = """
You extract a career timeline from Wikipedia lead text + Wikidata occupations.

Rules:
- Do NOT invent careers not present in occupations_input unless biography clearly supports a very close synonym.
- Clean/merge occupation titles ONLY when they are near-synonyms for the same specific role:
  "film actor" and "television actor" -> "actor". Do NOT merge occupations that represent genuinely
  different roles or levels of responsibility just because they're related: "mayor" is a specific
  elected office and must stay distinct from the generic "politician"; "businessperson" is distinct
  from "politician".
- Create exactly one stage in "stages" for every entry in occupations_clean - never omit one, even if
  the biography gives no explicit timing for that occupation.
- Only fill in start_year/end_year for a stage when the biography text gives concrete evidence of WHEN
  that specific role happened (an explicit year/period, or a clear narrative marker like "later" or
  "after retiring"). If there is no such evidence, leave start_year and end_year as null for that stage
  - but still include the stage itself with its occupation and best-available evidence.
- IMPORTANT: if the quote you use for "evidence" itself contains a specific year number, you MUST fill
  in start_year (and end_year if a range is given) from that same quote - do not leave start_year null
  when the year is already sitting in the evidence text you chose.
- The "evidence" field must be a real quote or close paraphrase from the biography that is directly
  ABOUT that occupation (e.g. describes them doing that job). Never attach evidence about an unrelated
  event to an occupation it doesn't describe - this includes lawsuits, awards, election results, personal
  life, death, AND any law, policy, or other consequence that resulted from their work but does not itself
  describe them performing that occupation.
- Order dated stages strictly chronologically by start_year, earliest first. Undated stages should be
  placed in the position implied by where they are discussed in the biography's narrative order.
- Only fill in start_year/end_year when a specific year number is written in the biography text
  for that exact event. NEVER use outside/background knowledge (e.g. a birth year you happen to
  know) to guess a year - leave start_year/end_year as null instead if the text doesn't state one.
  In particular, NEVER output a career-stage year that is simply the person's own birth or death year
  (positive or negated) unless the stage is literally about their birth or death.
- If the biography text marks a year as BC/BCE, use a NEGATIVE integer for it (e.g. "51 BC" -> -51).
  Do not apply a negative sign unless the text explicitly says BC/BCE for that specific date.
- Return ONLY valid JSON matching exactly this schema:

{
  "qid": "Q123",
  "occupations_input": ["..."],
  "occupations_clean": ["..."],
  "stages": [
    {"stage_order": 1, "occupation_clean": "...", "evidence": "...", "start_year": 1900, "end_year": 1910}
  ],
  "notes": "..."
}
"""

# Appended to the prompt when refining from a non-English biography. SYSTEM_RULES itself
# stays in English (an 8B model follows English instructions more reliably even when the
# source text is German) - only this note, and its instruction to the model, is language-
# specific.
_LANG_PROMPT_NOTE = {
    "en": "",
    "de": ("\nThe biography text below is written in German. Write occupations_clean, "
           "the evidence quotes, and notes in German too, matching the wording used in "
           "the biography text - do not translate them into English. Keep the JSON field "
           "names and overall structure exactly as specified above (in English).\n"),
}

# BC/BCE marker, by language - German Wikipedia writes "v. Chr." ("vor Christus"),
# not "BC"/"BCE".
_BC_MARKER_RE = {
    "en": re.compile(r"\bBCE?\b"),
    "de": re.compile(r"\bv\.\s?Chr\.?\b"),
}

def _ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": OLLAMA_TEMPERATURE, "num_predict": NUM_PREDICT_CAP}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()["response"]

_YEAR_WINDOW = 150  # chars of bio_lead searched around a stage's evidence text

# "1976 to 1979" / "1976-1979" / "1976–1979" (both years written in full)
_FULL_RANGE_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\s*(?:to|-|–|—)\s*(1[0-9]{3}|20[0-2][0-9])\b")
# German equivalent: "1976 bis 1979" (plus the same punctuation-only forms English uses)
_FULL_RANGE_RE_DE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\s*(?:bis|-|–|—)\s*(1[0-9]{3}|20[0-2][0-9])\b")
_FULL_RANGE_RE_BY_LANG = {"en": _FULL_RANGE_RE, "de": _FULL_RANGE_RE_DE}
# "1834–45" / "1834-45" (end year abbreviated to its last 2 digits, same century as the start)
# - punctuation-only, no language-specific wording, used as-is for every language.
_ABBREV_RANGE_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\s*(?:-|–|—)\s*([0-9]{2})\b")

def _year_in_text(year: int, text: str) -> bool:
    """True if `year` is written in `text` either as a plain 4-digit number, or as the
    end of an abbreviated range like "1834-45" (meaning 1845) - bios commonly abbreviate
    the second year of a same-century range this way, which a plain substring check on
    the full 4-digit form misses entirely.

    Uses word-boundary matching, not plain substring containment: a naive `"51" in text`
    check is true for text containing "1517" (since "51" is a literal substring of
    "1517"), which let a real fabricated year (51) falsely appear "supported" by an
    unrelated real year (1517) nearby. \\b ensures 51 only matches a genuine standalone
    "51", not a fragment embedded in a longer number."""
    year = abs(year)
    if re.search(rf"\b{year}\b", text):
        return True
    for m in _ABBREV_RANGE_RE.finditer(text):
        start, end_2digit = int(m.group(1)), int(m.group(2))
        end = (start // 100) * 100 + end_2digit
        if year in (start, end):
            return True
    return False

def _discard_unsupported_years(
    timeline: CareerTimeline, bio_lead: str, birth_year: Optional[int] = None, death_year: Optional[int] = None,
    lang: str = "en",
) -> CareerTimeline:
    """Null out a stage's start_year/end_year unless it's actually grounded near that
    stage's own evidence text, or it exactly matches the person's own birth/death year
    (almost never a legitimate career-stage year - see below).

    Two checks, both learned from real failures found by hand-checking pipeline output:

    1. Windowed proximity: the year must appear within _YEAR_WINDOW characters of where
       the stage's evidence text is located in bio_lead - not just anywhere in the bio.
       A stage's evidence field is sometimes a truncated paraphrase (e.g. "...mayor of
       Trondheim...") that omits a date clause immediately following it in the real text
       ("...from 1976 to 1979") - checking a window recovers that. But checking the whole
       bio_lead (an earlier version of this function) was too permissive: it let through
       a year that technically exists in the bio but describes a completely different,
       nearby-but-unrelated event (e.g. evidence about "Jewish education" sitting next to
       an unrelated sentence about a law passed in 1847).
    2. Birth/death-year exclusion: reject any candidate year equal to +/- the person's own
       birth or death year. The model has a specific, repeated failure mode of fabricating
       a stage year by negating a birth year it knows from training data (as if applying
       the BC/BCE rule to a person who was never BC/BCE) - since that digit genuinely
       appears in bio_lead's own birth/death parenthetical, the windowed/whole-bio checks
       alone can't catch it; this rule catches it directly regardless of proximity.
    3. BC/BCE sign correction: a negative year is CORRECTED to positive (not discarded)
       unless bio_lead literally contains "BC" or "BCE" anywhere. Hand-checking real
       output found this is a much bigger problem than #2 alone, and mostly unrelated to
       birth/death years: the model applies a spurious negative sign to completely
       ordinary, real, unrelated AD years (e.g. "he first used a camera in 1896" -> -1896,
       with no BC/BCE anywhere in the bio). Since the underlying digits are usually
       correct and well-evidenced, correcting the sign preserves real data that blind
       discarding would waste - the corrected value still has to pass every check below
       (birth/death/window) before being kept.
    """
    # Word-boundary match, not substring: a naive "BC" in bio_lead check is true for
    # any bio mentioning "CBC Radio" or "NBC television" - both contain "BC" as a
    # literal substring with nothing to do with BC/BCE dates. Marker is language-specific
    # (English "BC"/"BCE" vs German "v. Chr.").
    bio_has_bc = bool(_BC_MARKER_RE[lang].search(bio_lead))
    for stage in timeline.stages:
        evidence = stage.evidence or ""
        prefix = evidence.rstrip(".").rstrip("…")
        if prefix.endswith("..."):
            prefix = prefix[:-3]
        prefix = prefix.strip()[:60]
        idx = bio_lead.find(prefix) if len(prefix) >= 20 else -1
        if idx != -1:
            window = bio_lead[max(0, idx - _YEAR_WINDOW): idx + len(prefix) + _YEAR_WINDOW]
        else:
            window = evidence  # can't locate evidence in bio_lead - fall back to the evidence text itself

        for attr in ("start_year", "end_year"):
            year = getattr(stage, attr)
            if year is None:
                continue
            if year < 0 and not bio_has_bc:
                year = -year
                setattr(stage, attr, year)  # correct the sign; still re-validated below
            if birth_year is not None and abs(year) == abs(birth_year):
                setattr(stage, attr, None)
            elif death_year is not None and abs(year) == abs(death_year):
                setattr(stage, attr, None)
            elif birth_year is not None and year > 0 and year < birth_year:
                # can't have worked before being born (e.g. a posthumous republication
                # date mistaken for when they were active)
                setattr(stage, attr, None)
            elif death_year is not None and year > 0 and year > death_year:
                # can't have worked after dying (e.g. a later edition of their book
                # published after their death)
                setattr(stage, attr, None)
            elif not _year_in_text(year, window):
                setattr(stage, attr, None)
    return timeline

def _fill_missing_years_from_evidence(timeline: CareerTimeline, lang: str = "en") -> CareerTimeline:
    """Backfill start_year/end_year from a stage's OWN evidence text when the model left
    them null despite the year being right there - an instruction ("fill in the year if
    it's in your evidence quote") the 8B model does not reliably follow even when stated
    explicitly in the prompt. Scoped to only the stage's own evidence (not bio_lead
    broadly), so this is high-confidence: the evidence is already established to be about
    this occupation, so any year within it is safe to attach to this stage."""
    for stage in timeline.stages:
        if stage.start_year is not None or stage.end_year is not None:
            continue
        evidence = stage.evidence or ""
        m = _FULL_RANGE_RE_BY_LANG[lang].search(evidence)
        if m:
            stage.start_year, stage.end_year = int(m.group(1)), int(m.group(2))
            continue
        m = _ABBREV_RANGE_RE.search(evidence)
        if m:
            start, end_2digit = int(m.group(1)), int(m.group(2))
            stage.start_year = start
            stage.end_year = (start // 100) * 100 + end_2digit
            continue
        m = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", evidence)
        if m:
            stage.start_year = int(m.group(1))
    return timeline

def _ensure_stage_per_occupation(timeline: CareerTimeline) -> CareerTimeline:
    """Guarantee every entry in occupations_clean has at least one matching stage,
    regardless of whether the model actually emitted one - the "always create one stage
    per occupation" prompt rule is not reliably followed by the 8B model. Any occupation
    with no stage gets a minimal undated placeholder appended (empty evidence, so it's
    clearly distinguishable from a real model-grounded stage), rather than silently
    vanishing from career_timeline."""
    covered = {s.occupation_clean for s in timeline.stages}
    next_order = len(timeline.stages) + 1
    for occ in timeline.occupations_clean:
        if occ not in covered:
            timeline.stages.append(CareerStage(
                stage_order=next_order, occupation_clean=occ, evidence="",
                start_year=None, end_year=None,
            ))
            next_order += 1
    return timeline

def _sort_stages_chronologically(timeline: CareerTimeline) -> CareerTimeline:
    """Re-order stages by start_year regardless of what the LLM emitted, since an
    8B model frequently gets chronological ordering wrong even when it reports
    correct years. Undated stages keep their relative LLM-assigned order and are
    placed after all dated stages."""
    dated = [s for s in timeline.stages if s.start_year is not None]
    undated = [s for s in timeline.stages if s.start_year is None]
    dated.sort(key=lambda s: s.start_year)

    ordered = dated + undated
    for i, stage in enumerate(ordered, start=1):
        stage.stage_order = i

    timeline.stages = ordered
    return timeline

def build_timeline_with_llm(
    qid: str,
    occupations: List[str],
    bio_lead: str,
    birth_year: Optional[int] = None,
    death_year: Optional[int] = None,
    lang: str = "en",
) -> CareerTimeline:
    prompt = f"""
{SYSTEM_RULES}
{_LANG_PROMPT_NOTE[lang]}
QID: {qid}

occupations_input:
{json.dumps(occupations, ensure_ascii=False)}

Wikipedia lead text:
\"\"\"{bio_lead}\"\"\"
"""

    last_error = None
    for _ in range(OLLAMA_MAX_RETRIES):
        try:
            raw = _ollama_generate(prompt).strip()
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            time.sleep(2)
            continue

        # try to extract JSON if model added extra text
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end+1]

        try:
            timeline = CareerTimeline.model_validate_json(raw)
            timeline = _fill_missing_years_from_evidence(timeline, lang=lang)
            timeline = _discard_unsupported_years(timeline, bio_lead, birth_year, death_year, lang=lang)
            timeline = _ensure_stage_per_occupation(timeline)
            return _sort_stages_chronologically(timeline)
        except (ValidationError, json.JSONDecodeError) as e:
            last_error = str(e)
            prompt += f"\n\nYour last output was invalid. Fix it and output ONLY valid JSON. Error: {last_error}\n"

    raise ValueError(f"Ollama failed to return valid JSON after retries: {last_error}")
