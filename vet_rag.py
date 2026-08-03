"""
vet_rag.py — RAG pipeline for PawPal+ veterinary records.

Ingests free-text veterinary notes/history, retrieves the passages relevant to
each clinical aspect (labs, diagnoses, prescriptions, diet, vaccines), and
produces a structured VisitSummary:

  - succinct summary of the visit
  - laboratory results + contraindications implied by those results
  - diagnosed conditions with wiki/reference links
  - recommended special food formulations with ordering links
  - prescriptions with dose, frequency, price estimate, and ordering links

Two execution paths:
  1. LLM path (preferred) — Claude (claude-opus-5) with a strict JSON schema
     via output_config.format, so the response is guaranteed-parseable JSON.
  2. Offline path — a heuristic regex/keyword extractor used automatically
     when the anthropic SDK is unavailable or no API credentials are set.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import quote_plus

import lab_interpreter

MODEL = "claude-opus-5"

# ---------------------------------------------------------------------------
# Structured result types
# ---------------------------------------------------------------------------


@dataclass
class LabResult:
    name: str
    value: str
    unit: str = ""
    reference_range: str = ""
    flag: str = "normal"  # "normal" | "high" | "low" | "abnormal"
    note: str = ""        # plain-language interpretation when out of range


@dataclass
class Contraindication:
    finding: str          # the lab result / condition driving the warning
    avoid: str            # what to avoid (drug class, food, activity)
    reason: str           # why it matters


@dataclass
class ConditionInfo:
    name: str
    summary: str = ""
    wiki_url: str = ""
    reference_url: str = ""


@dataclass
class FoodRecommendation:
    product: str
    reason: str = ""
    order_links: dict[str, str] = field(default_factory=dict)


@dataclass
class Prescription:
    name: str
    dose: str = ""                 # e.g. "5 mg" or "1 tablet (75 mg)"
    frequency: str = ""            # e.g. "twice daily", "q12h", "once monthly"
    duration: str = ""             # e.g. "14 days", "ongoing"
    purpose: str = ""
    price_estimate: str = ""       # e.g. "$25-40 / month"
    refills: str = ""
    pickup_required: bool = True
    order_links: dict[str, str] = field(default_factory=dict)


@dataclass
class VaccineRecord:
    name: str
    date_given: str = ""           # ISO date if known
    next_due: str = ""             # ISO date if known


@dataclass
class VisitSummary:
    pet_name: str = ""
    visit_date: str = ""
    reason_for_visit: str = ""
    summary: str = ""
    diagnoses: list[ConditionInfo] = field(default_factory=list)
    lab_results: list[LabResult] = field(default_factory=list)
    lab_summary: str = ""  # plain-language interpretation of the lab panel
    contraindications: list[Contraindication] = field(default_factory=list)
    foods: list[FoodRecommendation] = field(default_factory=list)
    prescriptions: list[Prescription] = field(default_factory=list)
    vaccines: list[VaccineRecord] = field(default_factory=list)
    follow_up: str = ""            # e.g. "Recheck bloodwork in 4 weeks"
    generated_by: str = "offline"  # "claude" | "offline"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Link builders — search URLs are stable and never 404
# ---------------------------------------------------------------------------


def wiki_link(condition: str) -> str:
    return f"https://en.wikipedia.org/wiki/Special:Search?search={quote_plus(condition)}"


def vet_reference_link(condition: str) -> str:
    return f"https://www.merckvetmanual.com/searchresults?query={quote_plus(condition)}"


def food_order_links(product: str) -> dict[str, str]:
    q = quote_plus(product)
    return {
        "Chewy": f"https://www.chewy.com/s?query={q}",
        "Amazon": f"https://www.amazon.com/s?k={q}",
    }


def pharmacy_order_links(drug: str) -> dict[str, str]:
    q = quote_plus(drug)
    return {
        "Chewy Pharmacy": f"https://www.chewy.com/s?query={q}",
        "PetMeds": f"https://www.1800petmeds.com/search?q={q}",
    }


def _apply_lab_interpretation(summary: VisitSummary) -> VisitSummary:
    """Re-derive lab flags/notes from value vs. reference range and set lab_summary.

    Interpretation is deterministic (lab_interpreter), so it does not depend on
    the LLM correctly flagging results — the model extracts values, this fills in
    the clinical reading.
    """
    if not summary.lab_results:
        return summary
    panel = lab_interpreter.interpret_findings([
        {"name": lr.name, "value": lr.value, "unit": lr.unit,
         "reference_range": lr.reference_range}
        for lr in summary.lab_results
    ])
    for lr, finding in zip(summary.lab_results, panel.findings):
        lr.flag = finding.flag
        lr.note = finding.note
    if not summary.lab_summary:
        summary.lab_summary = panel.summary
    return summary


def _attach_links(summary: VisitSummary) -> VisitSummary:
    """Fill in link fields for every diagnosis, food, and prescription."""
    for cond in summary.diagnoses:
        if not cond.wiki_url:
            cond.wiki_url = wiki_link(cond.name)
        if not cond.reference_url:
            cond.reference_url = vet_reference_link(cond.name)
    for food in summary.foods:
        if not food.order_links:
            food.order_links = food_order_links(food.product)
    for rx in summary.prescriptions:
        if not rx.order_links:
            rx.order_links = pharmacy_order_links(rx.name)
    return summary


# ---------------------------------------------------------------------------
# Retrieval — chunking + lexical (TF-IDF style) scoring, pure Python
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def chunk_notes(text: str, max_chars: int = 800) -> list[str]:
    """Split vet notes into retrievable chunks on section headers / blank lines.

    Keeps section headers ("Assessment:", "Plan:", "Labs:") attached to their
    content so a retrieved chunk carries its clinical context. Long sections
    are further split at sentence boundaries to stay under max_chars.
    """
    blocks = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            chunks.append(block)
            continue
        # split long blocks at line boundaries
        current = ""
        for line in block.splitlines():
            if len(current) + len(line) + 1 > max_chars and current:
                chunks.append(current.strip())
                current = ""
            current += line + "\n"
        if current.strip():
            chunks.append(current.strip())
    return chunks


class NotesIndex:
    """Tiny in-memory lexical index over vet-note chunks (TF-IDF scoring)."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self._chunk_tokens = [_tokenize(c) for c in chunks]
        self._doc_freq: dict[str, int] = {}
        for tokens in self._chunk_tokens:
            for tok in set(tokens):
                self._doc_freq[tok] = self._doc_freq.get(tok, 0) + 1

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Return up to k chunks ranked by TF-IDF relevance to the query."""
        n = len(self.chunks)
        if n == 0:
            return []
        q_tokens = _tokenize(query)
        scores: list[tuple[float, int]] = []
        for i, tokens in enumerate(self._chunk_tokens):
            if not tokens:
                continue
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            score = 0.0
            for tok in q_tokens:
                if tok in tf:
                    idf = math.log(1 + n / self._doc_freq.get(tok, 1))
                    score += (tf[tok] / len(tokens)) * idf
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:k]]


# The clinical aspects the summarizer cares about; each becomes a retrieval query.
_ASPECT_QUERIES = [
    "laboratory results bloodwork chemistry CBC values",
    "assessment diagnosis condition disease",
    "prescription medication dose frequency dispense refills",
    "diet food formulation prescription diet feeding",
    "vaccine vaccination booster due rabies",
    "plan follow up recheck instructions",
]


def build_context(notes: str, max_context_chars: int = 6000) -> str:
    """Assemble the retrieval context for the summarizer.

    Short notes are passed through whole. Long histories are reduced by
    retrieving the top chunks for each clinical aspect and deduplicating.
    """
    if len(notes) <= max_context_chars:
        return notes.strip()
    index = NotesIndex(chunk_notes(notes))
    seen: set[str] = set()
    picked: list[str] = []
    total = 0
    for query in _ASPECT_QUERIES:
        for chunk in index.retrieve(query, k=3):
            if chunk in seen:
                continue
            if total + len(chunk) > max_context_chars:
                continue
            seen.add(chunk)
            picked.append(chunk)
            total += len(chunk)
    return "\n\n".join(picked)


# ---------------------------------------------------------------------------
# LLM path — Claude with a strict JSON schema
# ---------------------------------------------------------------------------

_SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "pet_name", "visit_date", "reason_for_visit", "summary", "diagnoses",
        "lab_results", "contraindications", "foods", "prescriptions",
        "vaccines", "follow_up",
    ],
    "properties": {
        "pet_name": {"type": "string"},
        "visit_date": {"type": "string", "description": "ISO date if stated, else empty"},
        "reason_for_visit": {"type": "string"},
        "summary": {"type": "string", "description": "Succinct 2-4 sentence summary of the visit"},
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "summary"],
                "properties": {
                    "name": {"type": "string"},
                    "summary": {"type": "string", "description": "One-sentence plain-language explanation"},
                },
            },
        },
        "lab_results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "value", "unit", "reference_range", "flag"],
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                    "unit": {"type": "string"},
                    "reference_range": {"type": "string"},
                    "flag": {"type": "string", "enum": ["normal", "high", "low", "abnormal"]},
                },
            },
        },
        "contraindications": {
            "type": "array",
            "description": "Contraindications implied by the lab results or diagnoses",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["finding", "avoid", "reason"],
                "properties": {
                    "finding": {"type": "string"},
                    "avoid": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "foods": {
            "type": "array",
            "description": "Recommended special/prescription food formulations",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["product", "reason"],
                "properties": {
                    "product": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "prescriptions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "dose", "frequency", "duration", "purpose",
                             "price_estimate", "refills", "pickup_required"],
                "properties": {
                    "name": {"type": "string"},
                    "dose": {"type": "string"},
                    "frequency": {"type": "string"},
                    "duration": {"type": "string"},
                    "purpose": {"type": "string"},
                    "price_estimate": {"type": "string", "description": "Typical US retail price range, e.g. '$20-35 / month'; empty if unknown"},
                    "refills": {"type": "string"},
                    "pickup_required": {"type": "boolean"},
                },
            },
        },
        "vaccines": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "date_given", "next_due"],
                "properties": {
                    "name": {"type": "string"},
                    "date_given": {"type": "string"},
                    "next_due": {"type": "string"},
                },
            },
        },
        "follow_up": {"type": "string"},
    },
}

_SYSTEM_PROMPT = (
    "You are a veterinary records assistant inside a pet-care scheduling app. "
    "You are given excerpts from a pet's veterinary notes/history. Extract only "
    "what the notes support - never invent lab values, diagnoses, doses, or "
    "dates. Dates must be ISO format (YYYY-MM-DD) when derivable, else empty "
    "strings. For contraindications, reason from the lab results and diagnoses "
    "(e.g. elevated liver enzymes -> avoid hepatotoxic NSAIDs). Price estimates "
    "are typical US retail ranges and clearly approximate; leave empty when you "
    "are not reasonably confident. This assists a pet owner's scheduling - it "
    "is not medical advice."
)


def _summary_from_payload(data: dict, generated_by: str) -> VisitSummary:
    """Build a VisitSummary from a parsed JSON payload (LLM output)."""
    summary = VisitSummary(
        pet_name=data.get("pet_name", ""),
        visit_date=data.get("visit_date", ""),
        reason_for_visit=data.get("reason_for_visit", ""),
        summary=data.get("summary", ""),
        diagnoses=[ConditionInfo(name=d["name"], summary=d.get("summary", ""))
                   for d in data.get("diagnoses", [])],
        lab_results=[LabResult(**lr) for lr in data.get("lab_results", [])],
        contraindications=[Contraindication(**c) for c in data.get("contraindications", [])],
        foods=[FoodRecommendation(product=f["product"], reason=f.get("reason", ""))
               for f in data.get("foods", [])],
        prescriptions=[Prescription(
            name=p["name"], dose=p.get("dose", ""), frequency=p.get("frequency", ""),
            duration=p.get("duration", ""), purpose=p.get("purpose", ""),
            price_estimate=p.get("price_estimate", ""), refills=p.get("refills", ""),
            pickup_required=bool(p.get("pickup_required", True)),
        ) for p in data.get("prescriptions", [])],
        vaccines=[VaccineRecord(**v) for v in data.get("vaccines", [])],
        follow_up=data.get("follow_up", ""),
        generated_by=generated_by,
    )
    _apply_lab_interpretation(summary)
    return _attach_links(summary)


def _summarize_with_claude(notes: str, pet_name: Optional[str]) -> VisitSummary:
    import anthropic  # imported lazily so the offline path has no hard dependency

    client = anthropic.Anthropic()
    context = build_context(notes)
    user_prompt = (
        (f"The pet's name is {pet_name}.\n\n" if pet_name else "")
        + "Veterinary notes/history excerpts:\n\n<vet_notes>\n"
        + context
        + "\n</vet_notes>\n\nExtract the structured visit summary."
    )

    # Server-side refusal fallback is enabled by default so a safety-classifier
    # decline is transparently retried on Anthropic's recommended fallback model.
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=_SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": _SUMMARY_SCHEMA}},
        messages=[{"role": "user", "content": user_prompt}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to process these notes.")
    text = next(b.text for b in response.content if b.type == "text")
    return _summary_from_payload(json.loads(text), generated_by="claude")


# ---------------------------------------------------------------------------
# Offline fallback — heuristic extraction, no network required
# ---------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _normalize_date(raw: str) -> str:
    """Normalize a date to ISO (YYYY-MM-DD). Accepts YYYY-MM-DD, DD-Mon-YYYY, M/D/YYYY."""
    raw = raw.strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw)
    if m:
        return raw
    m = re.match(r"^(\d{1,2})-([A-Za-z]{3})[a-z]*-(\d{4})$", raw)
    if m:
        month = _MONTHS.get(m.group(2).lower())
        if month:
            return f"{m.group(3)}-{month:02d}-{int(m.group(1)):02d}"
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", raw)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return ""


_ANY_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}-[A-Za-z]{3}[a-z]*-\d{4}|\d{1,2}/\d{1,2}/\d{4}")

# Dose amount+form and a broad frequency phrase, found independently on a med line.
_DOSE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|chewable tablets?|tablets?|"
    r"caps?(?:ules?)?|chews?|injections?|tabs?))\b",
    re.IGNORECASE,
)
_FREQ_RE = re.compile(
    r"(once a month|once (?:a day|daily)|every other day|every \d+(?:-\d+)? hours?|"
    r"as needed|twice daily|three times daily|q\d+h|BID|TID|SID|QID|PRN|"
    r"monthly|weekly|daily)",
    re.IGNORECASE,
)
# drug names that appear without an explicit dose+frequency but should still be captured
_KNOWN_DRUGS = ("moxidectin", "proheart", "credelio", "trazodone", "nexgard",
                "heartgard", "simparica", "apoquel", "cytopoint", "gabapentin")


def _extract_meds(notes: str) -> list["Prescription"]:
    """Extract prescriptions from free-form medication lines.

    A line qualifies if it names a dose+form (or a known drug) and, ideally, a
    frequency phrase. The drug name is taken as the leading word(s) of the line.
    """
    meds: list[Prescription] = []
    seen: set[str] = set()
    for line in notes.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        dose_m = _DOSE_RE.search(stripped)
        freq_m = _FREQ_RE.search(stripped)
        known = next((d for d in _KNOWN_DRUGS if d in low), None)
        if not (dose_m or known) or not (freq_m or known):
            continue
        # name = leading alphabetic words, up to the first number, colon, or paren
        nm = re.match(r"^([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,2})", stripped)
        name = (nm.group(1).strip() if nm else (known or "")).strip()
        # drop a leading imperative verb ("Start Denamarin" -> "Denamarin")
        name = re.sub(r"^(?:start|give|administer|apply|begin|continue|dispense)\s+",
                      "", name, flags=re.IGNORECASE).strip()
        if not name or name.lower() in ("plan", "assessment"):
            name = (known or "").title()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        # PRN ("as needed"/fireworks) meds are on-demand — prefer that over a
        # clock frequency also present on the line, so the agent doesn't
        # schedule recurring doses for something only given as needed.
        if "as needed" in low or "prn" in low:
            frequency = "as needed"
        elif freq_m:
            frequency = freq_m.group(1).strip()
        else:
            frequency = "as directed"
        meds.append(Prescription(
            name=name,
            dose=dose_m.group(1).strip() if dose_m else "",
            frequency=frequency,
        ))
    return meds


_VACCINE_NAMES = ("rabies", "dhpp", "da2pp", "distemper", "parvo", "bordetella",
                  "leptospirosis", "lepto", "fvrcp", "felv", "influenza", "lyme")

_VACCINE_DISPLAY = {
    "rabies": "Rabies", "dhpp": "DHPP", "da2pp": "DA2PP", "distemper": "Distemper",
    "parvo": "Parvo", "bordetella": "Bordetella", "leptospirosis": "Leptospirosis",
    "lepto": "Leptospirosis", "fvrcp": "FVRCP", "felv": "FeLV",
    "influenza": "Influenza", "lyme": "Lyme",
}

_CONDITION_KEYWORDS = {
    "chronic kidney disease": "Chronic kidney disease (CKD)",
    "kidney disease": "Kidney disease",
    "renal insufficiency": "Renal insufficiency",
    "hyperthyroid": "Hyperthyroidism",
    "hypothyroid": "Hypothyroidism",
    "diabetes": "Diabetes mellitus",
    "pancreatitis": "Pancreatitis",
    "arthritis": "Osteoarthritis",
    "osteoarthritis": "Osteoarthritis",
    "dental disease": "Dental disease",
    "periodontal": "Periodontal disease",
    "otitis": "Otitis (ear infection)",
    "dermatitis": "Dermatitis",
    "allerg": "Allergies",
    "obesity": "Obesity",
    "overweight": "Overweight",
    "heart murmur": "Heart murmur",
    "cardiomyopathy": "Cardiomyopathy",
    "urinary tract infection": "Urinary tract infection (UTI)",
    "cystitis": "Cystitis",
    "hepatitis": "Hepatitis",
    "liver disease": "Liver disease",
    "hepatopathy": "Hepatopathy",
    "anemia": "Anemia",
    "heartworm": "Heartworm disease",
}

_FOOD_BRANDS = ("hill's", "hills", "royal canin", "purina", "k/d", "c/d", "i/d",
                "z/d", "w/d", "urinary so", "renal support", "pro plan",
                "hydrolyzed", "prescription diet")

# contraindication rules keyed on (lab-name substring, flag)
_CONTRA_RULES = [
    (("alt", "ast", "alp", "alkp", "ggt", "bilirubin", "tbil"), "high",
     "hepatotoxic drugs (e.g. certain NSAIDs)",
     "Elevated liver values - drugs metabolized by the liver may worsen damage."),
    (("bun", "creatinine", "crea", "sdma", "phosphorus", "phos"), "high",
     "NSAIDs and nephrotoxic drugs; high-phosphorus/high-protein diets",
     "Elevated kidney values - NSAIDs reduce renal blood flow and phosphorus accelerates CKD."),
    (("glucose", "glu"), "high",
     "high-carbohydrate treats and foods",
     "Elevated glucose - simple carbohydrates worsen glycemic control."),
    (("hct", "hgb", "rbc"), "low",
     "strenuous activity until rechecked",
     "Low red-cell values (anemia) - reduced oxygen-carrying capacity."),
    (("plt", "platelets"), "low",
     "drugs that impair clotting (aspirin, other NSAIDs)",
     "Low platelets - increased bleeding risk."),
]


def _extract_offline(notes: str, pet_name: Optional[str]) -> VisitSummary:
    """Heuristic extraction used when the Claude API is unavailable."""
    lines = notes.splitlines()
    lower = notes.lower()

    # --- labs (parsed and interpreted against reference ranges) ---
    panel = lab_interpreter.interpret_lab_panel(notes)
    labs: list[LabResult] = [
        LabResult(name=f.name, value=f.value, unit=f.unit,
                  reference_range=f.reference_range, flag=f.flag, note=f.note)
        for f in panel.findings
    ]
    lab_summary = panel.summary

    # --- contraindications from lab flags ---
    contras: list[Contraindication] = []
    for lab in labs:
        for keys, flag, avoid, reason in _CONTRA_RULES:
            if lab.flag == flag and any(k in lab.name.lower() for k in keys):
                contras.append(Contraindication(
                    finding=f"{lab.name} {lab.value} {lab.unit} ({lab.flag})".strip(),
                    avoid=avoid, reason=reason))
                break

    # --- diagnoses ---
    diagnoses: list[ConditionInfo] = []
    seen_conditions: set[str] = set()
    for keyword, label in _CONDITION_KEYWORDS.items():
        if keyword in lower and label not in seen_conditions:
            seen_conditions.add(label)
            diagnoses.append(ConditionInfo(name=label))

    # --- prescriptions ---
    prescriptions: list[Prescription] = _extract_meds(notes)

    # --- vaccines (name + due date on the same line, any date format) ---
    vaccines: list[VaccineRecord] = []
    seen_vac: set[str] = set()
    for vac in _VACCINE_NAMES:
        display = _VACCINE_DISPLAY.get(vac, vac.title())
        if display in seen_vac:
            continue
        for line in lines:
            if vac in line.lower():
                due = ""
                m = _ANY_DATE_RE.search(line)
                if m:
                    due = _normalize_date(m.group(0))
                seen_vac.add(display)
                vaccines.append(VaccineRecord(name=display, next_due=due))
                break

    # --- food recommendations ---
    foods: list[FoodRecommendation] = []
    for line in lines:
        line_l = line.lower()
        if any(b in line_l for b in _FOOD_BRANDS):
            foods.append(FoodRecommendation(product=line.strip(" -•\t"),
                                            reason="Recommended in visit notes"))
            break

    # --- visit date / follow-up ---
    visit_date = ""
    anchor = re.search(r"(?:most recent visit date|visit date|reported|sample collected)[:\s]*"
                       r"(\d{4}-\d{2}-\d{2}|\d{1,2}-[A-Za-z]{3}[a-z]*-\d{4}|\d{1,2}/\d{1,2}/\d{4})",
                       notes, re.IGNORECASE)
    m = anchor or _ANY_DATE_RE.search(notes)
    if m:
        visit_date = _normalize_date(m.group(1) if anchor else m.group(0))
    follow_up = ""
    m = re.search(r"(?:recheck|follow.?up)[^.\n]*", notes, re.IGNORECASE)
    if m:
        follow_up = m.group(0).strip()

    abnormal = [f"{l.name} {l.flag}" for l in labs if l.flag in ("high", "low", "abnormal")]
    summary_bits = []
    if diagnoses:
        summary_bits.append("Diagnosed: " + ", ".join(d.name for d in diagnoses) + ".")
    if abnormal:
        summary_bits.append("Abnormal labs: " + ", ".join(abnormal) + ".")
    elif labs:
        summary_bits.append("Laboratory panel within normal limits.")
    if prescriptions:
        summary_bits.append("Prescribed: " + ", ".join(p.name for p in prescriptions) + ".")
    if follow_up:
        summary_bits.append(follow_up + ".")

    return _attach_links(VisitSummary(
        pet_name=pet_name or "",
        visit_date=visit_date,
        reason_for_visit="",
        summary=" ".join(summary_bits) or "No structured details could be extracted offline.",
        diagnoses=diagnoses,
        lab_results=labs,
        lab_summary=lab_summary,
        contraindications=contras,
        foods=foods,
        prescriptions=prescriptions,
        vaccines=vaccines,
        follow_up=follow_up,
        generated_by="offline",
    ))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_pdf_text(file_obj) -> str:
    """Extract text from an uploaded PDF file object using pypdf, if installed.

    Returns "" (and never raises) when pypdf is unavailable or the file can't be
    read, so the UI can fall back to manual paste.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(file_obj)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def claude_available() -> bool:
    """True when the anthropic SDK is importable and credentials are likely set."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def summarize_visit(notes: str, pet_name: Optional[str] = None) -> VisitSummary:
    """Summarize veterinary notes into a structured VisitSummary.

    Uses Claude when credentials are available; otherwise falls back to the
    offline heuristic extractor. Also falls back if the API call fails, so the
    app never hard-crashes on a network or auth problem.
    """
    if claude_available():
        try:
            return _summarize_with_claude(notes, pet_name)
        except Exception:
            pass  # fall through to offline extraction
    return _extract_offline(notes, pet_name)
