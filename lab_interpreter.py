"""
lab_interpreter.py — laboratory result interpretation & summarization.

Veterinary lab reports (Antech/IDEXX/VCA style) list a value and a reference
range but usually no HIGH/LOW flag — the reader is expected to compare them.
This module does that deterministically:

  - parses tabular lab rows ("ALT (SGPT) 35 12 - 118 IU/L") and reference
    ranges of every common shape ("5 - 7.4", "<14", ">2", "NEG TO 1+");
  - classifies each value against its range as normal / high / low / abnormal;
  - attaches a plain-language clinical note to out-of-range analytes;
  - detects reassuring screening results (parasites, heartworm, tick-borne);
  - rolls the whole panel up into a readable summary.

Interpretation is intentionally rule-based, not model-based: it is exact,
explainable, and unit-testable, and it runs the same whether or not the Claude
API is available. The LLM extracts raw values; this module interprets them.

Nothing here is medical advice — it summarizes what the report already states.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LabFinding:
    name: str
    value: str
    unit: str = ""
    reference_range: str = ""
    flag: str = "normal"   # "normal" | "high" | "low" | "abnormal" | "unknown"
    note: str = ""         # plain-language interpretation when out of range


@dataclass
class LabPanel:
    findings: list[LabFinding] = field(default_factory=list)
    summary: str = ""

    def abnormal(self) -> list[LabFinding]:
        return [f for f in self.findings if f.flag in ("high", "low", "abnormal")]


# ---------------------------------------------------------------------------
# Reference-range and value parsing
# ---------------------------------------------------------------------------

_NUM = r"-?\d+(?:\.\d+)?"
_NUMERIC_RE = re.compile(rf"^{_NUM}$")
_VALUE_RANGE_RE = re.compile(rf"^{_NUM}\s*[-–]\s*{_NUM}$")   # "0-1", "11-20"
_VALUE_PLUS_RE = re.compile(r"^\d+\+$")                       # "1+", "2+"
_BOUND_RE = re.compile(r"^[<>]\s*" + _NUM + r"$")             # "<14", ">2"


def parse_reference_range(ref: str) -> tuple[Optional[float], Optional[float], bool]:
    """Return (low, high, is_numeric).

    Either bound may be None (open-ended). is_numeric is False for qualitative
    ranges like "Negative" or "None seen".
    """
    ref = (ref or "").strip()
    m = re.match(rf"^<\s*({_NUM})$", ref)
    if m:
        return None, float(m.group(1)), True
    m = re.match(rf"^>\s*({_NUM})$", ref)
    if m:
        return float(m.group(1)), None, True
    m = re.match(rf"^({_NUM})\s*[-–]\s*({_NUM})$", ref)
    if m:
        return float(m.group(1)), float(m.group(2)), True
    return None, None, False


def parse_numeric_value(value: str) -> Optional[tuple[float, float]]:
    """Return (low, high) for a numeric value or numeric value-range, else None.

    A single number returns (n, n); a range like "0-1" returns (0, 1).
    Qualitative values ("1+", "NEGATIVE") return None.
    """
    value = (value or "").strip()
    m = re.match(rf"^({_NUM})$", value)
    if m:
        n = float(m.group(1))
        return n, n
    m = re.match(rf"^({_NUM})\s*[-–]\s*({_NUM})$", value)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


_NEGATIVE_TERMS = {
    "negative", "neg", "none seen", "none", "not detected",
    "no antigen detected", "adequate", "clear", "normal",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def classify_qualitative(value: str, ref: str) -> str:
    """Classify a non-numeric result (e.g. urine protein, screening tests)."""
    v, r = _norm(value), _norm(ref)
    if not r:
        return "unknown"
    if v == r:
        return "normal"
    # ref accepts up to N+ (e.g. bilirubin "NEG TO 1+")
    m = re.match(r"neg(?:ative)? to (\d+)\+", r)
    if m:
        mv = re.match(r"(\d+)\+", v)
        if v in _NEGATIVE_TERMS:
            return "normal"
        if mv:
            return "normal" if int(mv.group(1)) <= int(m.group(1)) else "abnormal"
        return "unknown"
    # ref is negative/none — any positive finding is abnormal
    if r in _NEGATIVE_TERMS:
        if v in _NEGATIVE_TERMS:
            return "normal"
        if _VALUE_PLUS_RE.match(v) or "positive" in v or "trace" in v or "detected" in v:
            return "abnormal"
    return "unknown"


def classify(value: str, ref: str) -> str:
    """Classify a value against its reference range."""
    vnum = parse_numeric_value(value)
    lo, hi, numeric_ref = parse_reference_range(ref)
    if vnum is not None and numeric_ref:
        vlo, vhi = vnum
        if lo is not None and vlo < lo:
            return "low"
        if hi is not None and vhi > hi:
            return "high"
        return "normal"
    return classify_qualitative(value, ref)


# ---------------------------------------------------------------------------
# Clinical notes for out-of-range analytes (owner-facing, not diagnostic)
# ---------------------------------------------------------------------------

# aliases -> canonical key used in _CLINICAL_NOTES
_ALIASES = {
    "alt": "alt", "alt sgpt": "alt", "ast": "ast", "ast sgot": "ast",
    "alk phos": "alp", "alkp": "alp", "alp": "alp", "ggt": "ggt",
    "bun": "bun", "urea": "bun", "creatinine": "creatinine", "crea": "creatinine",
    "sdma": "sdma", "glucose": "glucose", "glu": "glucose",
    "phosphorus": "phosphorus", "phos": "phosphorus",
    "calcium": "calcium", "ca": "calcium",
    "potassium": "potassium", "k": "potassium",
    "sodium": "sodium", "na": "sodium",
    "albumin": "albumin", "alb": "albumin",
    "total protein": "total_protein",
    "wbc": "wbc", "hct": "hct", "hgb": "hgb", "hemoglobin": "hgb",
    "platelet count": "platelets", "platelets": "platelets", "plt": "platelets",
    "protein": "urine_protein", "specific gravity": "usg",
    "bilirubin": "bilirubin", "tbil": "bilirubin",
    "t4": "t4", "cholesterol": "cholesterol",
}

_CLINICAL_NOTES = {
    "alt": {"high": "elevated liver enzyme; can indicate hepatocellular (liver) injury"},
    "ast": {"high": "can indicate liver or muscle injury"},
    "alp": {"high": "can reflect cholestasis, steroids, or bone activity"},
    "ggt": {"high": "supports a hepatobiliary (liver/gallbladder) origin"},
    "bilirubin": {"high": "can indicate cholestasis or red-cell breakdown"},
    "bun": {"high": "can indicate reduced kidney function or dehydration",
            "low": "usually not clinically significant"},
    "creatinine": {"high": "supports reduced kidney function",
                   "low": "usually not concerning (often low muscle mass)"},
    "sdma": {"high": "an early, sensitive marker of reduced kidney function"},
    "glucose": {"high": "can indicate diabetes or a stress response",
                "low": "can cause weakness or seizures if very low"},
    "phosphorus": {"high": "can accompany kidney disease"},
    "calcium": {"high": "warrants investigation (parathyroid, neoplasia)",
                "low": "can cause tremors or weakness"},
    "potassium": {"high": "can affect heart rhythm", "low": "can cause muscle weakness"},
    "sodium": {"high": "often reflects dehydration", "low": "can reflect fluid balance issues"},
    "albumin": {"low": "can reflect GI/renal loss or reduced liver production",
                "high": "usually reflects dehydration"},
    "total_protein": {"low": "can reflect protein loss", "high": "often dehydration or inflammation"},
    "wbc": {"high": "often indicates inflammation or infection",
            "low": "can indicate marrow suppression or overwhelming infection"},
    "hct": {"low": "anemia — reduced oxygen-carrying capacity",
            "high": "often dehydration or polycythemia"},
    "hgb": {"low": "anemia", "high": "often dehydration"},
    "platelets": {"low": "increased bleeding risk", "high": "usually a reactive change"},
    "urine_protein": {"abnormal": "proteinuria — protein in the urine; "
                                  "microalbuminuria testing / renal follow-up may be warranted"},
    "usg": {"low": "dilute urine — may indicate reduced concentrating ability"},
    "cholesterol": {"high": "can accompany endocrine disease (e.g. hypothyroidism)"},
    "t4": {"high": "can indicate hyperthyroidism", "low": "can indicate hypothyroidism"},
}


def _canonical(name: str) -> Optional[str]:
    n = re.sub(r"\([^)]*\)", "", name).lower()  # drop parentheticals like "(SGPT)"
    n = re.sub(r"\s+", " ", n).strip()
    return _ALIASES.get(n)


def clinical_note(name: str, flag: str) -> str:
    key = _canonical(name)
    if not key:
        return ""
    return _CLINICAL_NOTES.get(key, {}).get(flag, "")


# ---------------------------------------------------------------------------
# Row parsing from raw report text
# ---------------------------------------------------------------------------

_KNOWN_ANALYTES = set(_ALIASES.keys())


def _is_value_token(tok: str) -> bool:
    return bool(_NUMERIC_RE.match(tok) or _VALUE_RANGE_RE.match(tok) or _VALUE_PLUS_RE.match(tok))


_FLAG_WORDS = {"high": "high", "h": "high", "low": "low", "l": "low", "abnormal": "abnormal"}
_REF_NOISE = {"ref", "reference", "normal", ""}

# instruction / medication phrasing that must never be parsed as a lab row
_MED_HINT_RE = re.compile(
    r"\b(give|by mouth|as needed|once a|chewable|tablet|capsule|injection|"
    r"dispense|refill|prn|q\d+h|twice daily|three times|for fireworks|with food)\b",
    re.IGNORECASE,
)


def _consume_ref_and_unit(rest: list[str]) -> tuple[str, str]:
    """Scan the tokens after the value for a reference range and unit.

    Handles both column orders — value·range·unit (Antech tabular) and
    value·unit·(ref range) (free-text) — plus parenthesized "(ref 10-125)".
    """
    ref = ""
    unit_parts: list[str] = []
    i = 0
    while i < len(rest):
        tok = rest[i].strip("()")
        if not ref and _BOUND_RE.match(tok):                       # "<14" / ">2"
            ref = tok
            i += 1
            continue
        if not ref and _VALUE_RANGE_RE.match(tok):                 # "0-3"
            lo, hi = re.split(r"[-–]", tok, maxsplit=1)
            ref = f"{lo.strip()} - {hi.strip()}"
            i += 1
            continue
        if (not ref and _NUMERIC_RE.match(tok) and i + 2 < len(rest)
                and rest[i + 1] in ("-", "–") and _NUMERIC_RE.match(rest[i + 2])):  # "5 - 7.4"
            ref = f"{tok} - {rest[i + 2]}"
            i += 3
            continue
        if tok.lower() not in _REF_NOISE:
            unit_parts.append(tok)
        i += 1
    if not ref and unit_parts:
        # no numeric range found — treat the remainder as a qualitative reference
        return " ".join(unit_parts), ""
    return ref, " ".join(unit_parts)


def parse_lab_line(line: str) -> Optional[LabFinding]:
    """Parse one lab row into a classified LabFinding, or None.

    Handles tabular rows ("ALT (SGPT) 35 12 - 118 IU/L") and free-text rows
    with an explicit flag ("ALT 210 U/L (ref 10-125) HIGH"). Rejects non-lab
    lines (weights, dates, medication amounts) by requiring either a numeric
    reference range or a recognized analyte name.
    """
    line = line.strip()
    # medication/instruction lines can contain dose ranges ("every 8-12 hours")
    # that look like reference ranges — exclude them up front.
    if _MED_HINT_RE.search(line):
        return None
    # address / footer lines (e.g. "2917 Old US 231 South, Lafayette, IN 47909 |
    # 765 474 - 2454") can look like a value + range; the "|" separator and a
    # phone number never appear in a real lab row.
    if "|" in line:
        return None
    # pull a trailing explicit flag word off first (e.g. "... HIGH")
    explicit_flag = ""
    # require whitespace (or start) before the flag so unit letters like the
    # "L" in "IU/L" or "mg/dL" are never mistaken for a LOW flag
    fm = re.search(r"(?:^|(?<=\s))(HIGH|LOW|ABNORMAL|H|L)\s*$", line)
    if fm:
        explicit_flag = _FLAG_WORDS.get(fm.group(1).lower(), "")
        line = line[:fm.start()].rstrip()

    tokens = line.split()
    if len(tokens) < 2:
        return None

    idx = next((i for i in range(1, len(tokens)) if _is_value_token(tokens[i])), None)
    if idx is None:
        return None

    name = " ".join(tokens[:idx]).strip(" :")
    # analyte names are short and start with a letter — a leading digit means
    # the "name" is really a street number / address, not a lab test.
    if not name or not name[0].isalpha() or len(name.split()) > 5:
        return None
    value = tokens[idx]
    ref, unit = _consume_ref_and_unit(tokens[idx + 1:])

    _, _, numeric_ref = parse_reference_range(ref)
    flag = classify(value, ref)
    if flag in ("unknown", "normal") and explicit_flag:
        flag = explicit_flag  # trust the report's own flag when we can't derive one

    known = _canonical(name) is not None
    if not numeric_ref and flag == "unknown" and not known and not explicit_flag:
        return None

    return LabFinding(
        name=name, value=value, unit=unit.strip(), reference_range=ref,
        flag=flag, note=clinical_note(name, flag),
    )


# ---------------------------------------------------------------------------
# Screening tests (qualitative reassurances that add clinical context)
# ---------------------------------------------------------------------------

_SCREENINGS = [
    ("Ova & Parasite", ("ova & parasite", "ova and parasite")),
    ("Giardia", ("giardia",)),
    ("Heartworm antigen", ("heartworm",)),
    ("Borrelia (Lyme)", ("borrelia",)),
    ("Ehrlichia", ("ehrlichia",)),
    ("Anaplasma", ("anaplasma",)),
]


def detect_screenings(text: str) -> list[LabFinding]:
    """Detect qualitative infectious-disease/parasite screens and their result."""
    findings: list[LabFinding] = []
    seen: set[str] = set()
    for line in text.splitlines():
        low = line.lower()
        for label, keys in _SCREENINGS:
            if label in seen:
                continue
            if any(k in low for k in keys):
                negative = any(neg in low for neg in
                               ("negative", "none seen", "no antigen detected", "not detected"))
                positive = any(pos in low for pos in
                               ("positive", "detected", "seen")) and not negative
                if not (negative or positive):
                    continue
                seen.add(label)
                findings.append(LabFinding(
                    name=label, value="Negative" if negative else "Positive",
                    reference_range="Negative",
                    flag="normal" if negative else "abnormal",
                    note="" if negative else "positive screen — discuss with your vet",
                ))
    return findings


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def summarize_findings(findings: list[LabFinding]) -> str:
    """Roll a set of findings into an owner-facing plain-language summary."""
    if not findings:
        return "No interpretable laboratory values were found."
    abnormal = [f for f in findings if f.flag in ("high", "low", "abnormal")]
    total = len(findings)
    parts = [f"{total - len(abnormal)} of {total} interpreted results within normal limits."]
    if abnormal:
        listed = "; ".join(
            f"{f.name} {f.value}{(' ' + f.unit) if f.unit else ''} "
            f"({f.flag}{', ref ' + f.reference_range if f.reference_range else ''})"
            for f in abnormal
        )
        parts.append(f"Out of range: {listed}.")
        notes = [f"{f.name}: {f.note}." for f in abnormal if f.note]
        if notes:
            parts.append(" ".join(notes))
    else:
        parts.append("All interpreted values are within their reference ranges.")
    return " ".join(parts)


def interpret_lab_panel(text: str) -> LabPanel:
    """Parse and interpret every lab row + screening result in raw report text."""
    findings: list[LabFinding] = []
    for line in text.splitlines():
        f = parse_lab_line(line)
        if f is not None:
            findings.append(f)
    findings.extend(detect_screenings(text))
    return LabPanel(findings=findings, summary=summarize_findings(findings))


def interpret_findings(rows: list[dict]) -> LabPanel:
    """Re-interpret already-extracted lab rows (e.g. from the LLM).

    Each row needs at least name/value/reference_range; unit is optional. The
    flag and note are recomputed deterministically from value vs. range, so the
    interpretation never depends on the model guessing the flag correctly.
    """
    findings: list[LabFinding] = []
    for r in rows:
        name = r.get("name", "")
        value = str(r.get("value", ""))
        ref = r.get("reference_range", "")
        flag = classify(value, ref)
        findings.append(LabFinding(
            name=name, value=value, unit=r.get("unit", ""),
            reference_range=ref, flag=flag, note=clinical_note(name, flag),
        ))
    return LabPanel(findings=findings, summary=summarize_findings(findings))
