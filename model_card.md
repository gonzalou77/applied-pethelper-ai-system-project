# Model Card — PawPal+ AI Vet-Records System

This card documents the responsible-use considerations for the AI features added to PawPal+: the RAG vet-notes summarizer (`vet_rag.py`), the deterministic lab interpreter (`lab_interpreter.py`), and the scheduling agent (`vet_agent.py`).

**Intended use:** help a pet owner understand their own pet's veterinary records and stay organized with medication, vaccine, and follow-up scheduling. **Not intended use:** clinical diagnosis, treatment decisions, or a substitute for a veterinarian.

---

## 1) Limitations and biases

**System limitations**

- **Not medical advice.** The summarizer and interpreter restate and organize what a report already says. They cannot examine the animal, weigh clinical context, or catch an error the vet's own report contains.
- **Extraction is imperfect on messy input.** The offline heuristic extractor captures approximate drug names (e.g. "Credelio K9" → "Credelio K") and can miss meds phrased in ways its regexes don't anticipate. The Claude path is stronger but still bounded by whatever text was successfully pulled from the PDF (`pypdf` extraction quality varies by scan).
- **Lexical retrieval, not semantic.** The RAG retriever uses TF-IDF, so a passage that uses a pure synonym of the query terms can be missed. This is an acceptable trade for single-patient documents but would matter more at larger scale.
- **Finite, hand-curated clinical rules.** The lab interpreter's reference-range logic is exact, but its contraindication rules and clinical-note dictionary cover only common analytes. An out-of-range value with no rule gets a flag but no interpretive note.
- **Format and unit assumptions.** Parsing was validated against Antech/VCA-style reports (US units). A lab from a different provider, a different unit system, or an unusual layout may parse incompletely.
- **No temporal reasoning across visits.** Each record is interpreted on its own; the system does not trend a value across multiple reports to distinguish "stable" from "worsening."

**Potential biases**

- **Data/format bias.** The heuristics and tests are tuned to the specific report formats I had access to, so accuracy is highest for those and lower for under-represented layouts.
- **Species/breed coverage.** Reference ranges and rules are dog-oriented (the sample data was canine); cats and other species are under-covered.
- **Commercial link bias.** Ordering links point to a fixed set of retailers/pharmacies (Chewy, Amazon, PetMeds); they are convenience links, not endorsements or price comparisons.
- **Language/localization.** English-only, US date and unit conventions.

---

## 2) Misuse potential and mitigations

| Misuse risk | Mitigation in the system |
| --- | --- |
| **Treated as a diagnosis** instead of an organizer | Explicit "not medical advice" framing in the summarizer's system prompt and UI; a human-approval gate before the agent acts on anything. |
| **Autonomous medication scheduling errors** (e.g. scheduling a PRN drug as a daily dose, or dosing without owner awareness) | The agent only ever *proposes* tasks; the owner approves the care plan before it is applied. PRN ("as needed") meds are never auto-scheduled as recurring doses. Medication schedules are copied from the vet's stated frequency, not invented. |
| **Silently altering the owner's real schedule** | The agent's `sync()` only touches tasks it created (`source == "agent"`, a vet category); owner-created tasks are never modified or deleted. Every agent action is written to an audit log with a reason the owner can review. |
| **Fabricated clinical detail** (a model "hallucinating" a value, dose, or date) | Lab flags are computed deterministically from value vs. range, not guessed by the model. The LLM call uses a strict JSON schema, and the system prompt instructs it to extract only what the notes support. |
| **Exposure of others' medical/PII data** | The app is a local, single-user Streamlit tool; records stay on the user's machine. When the Claude path is used, only the retrieved excerpts are sent, and server-side refusal fallback is enabled. Users should only upload records for pets they are responsible for. |
| **Over-reliance leading to skipped vet visits** | Follow-up and vaccine-booster tasks are surfaced as scheduling items, and the tool consistently frames itself as an assistant to, not a replacement for, veterinary care. |

The strongest single mitigation is architectural: **the AI never takes an irreversible or clinical action on its own.** It extracts, interprets, and proposes; a human confirms.

---

## 3) What surprised me while testing reliability

- **Real records broke assumptions my synthetic tests never touched.** Everything passed on hand-written sample notes, then the actual VCA/Antech PDFs immediately exposed parsing bugs: a medication instruction line ("give every **8-12** hours") was misread as a lab because `8-12` looks like a reference range, and the single-letter `L`/`H` flag matcher caught the `L` in units like `IU/L` and `mg/dL`, flagging normal results as LOW. It was a concrete lesson that **synthetic fixtures validate the code you imagined, not the inputs you'll actually get.**
- **Determinism made "reliability" measurable.** Because lab interpretation is rule-based, I could assert exact flags on a real 40-value panel and know the interpreter was right — versus an LLM output I'd have to spot-check. Moving the clinically important step *out* of the model made the system easier to trust, not harder.
- **The agent's idempotency was reassuring to verify.** I expected the "continuous monitoring" loop to be fragile, but asserting that running `sync()` twice produces zero actions the second time made the self-healing behavior provably safe rather than hopeful.
- **The offline fallback was more capable than expected** on structured tabular reports, yet noticeably weaker on free-text drug phrasing — a clean illustration of where rules win (structured data) and where the LLM earns its place (messy natural language).

---

## 4) Collaboration with AI during this project

I built this project in close collaboration with an AI coding assistant, using it to scaffold modules, propose the RAG/agent architecture, and write and iterate on tests. I directed the design and reviewed every change; the AI accelerated implementation and surfaced options I then accepted or rejected.

**A helpful suggestion.** When I described the need for laboratory-result interpretation, the AI proposed making that step **deterministic and rule-based rather than model-based** — the LLM extracts raw values, and a separate `lab_interpreter` module compares each value to its reference range. This turned out to be the best design decision in the project: the most clinically important logic became exact, explainable, and fully unit-testable, and it worked identically with or without an API key. It also made the surprising real-data bugs (above) *findable*, because the behavior was pinned down by assertions instead of hidden in a model response.

**A flawed suggestion.** Early on, the AI's offline lab parser (and its tests) assumed lab lines would look like the synthetic format `ALT 210 U/L (ref 10-125) HIGH` — value, then unit, then an explicit HIGH/LOW flag. That assumption was wrong for real reports, which are tabular (`ALT (SGPT) 35 12 - 118 IU/L`) with **no flag at all** and a different column order. The AI-written regex both failed to parse the real format and, once I forced the issue with actual records, produced false positives (reading `IU/L`'s `L` as a LOW flag and a med line's `8-12` as a range). The fix required rejecting instruction lines, capping analyte-name length, and requiring whitespace before a flag letter — each backed by a new regression test. The lesson: an AI suggestion that looks correct against the examples it was given can still be wrong about the real world, so I treated its output as a draft to be validated against genuine data, not as a finished answer.
