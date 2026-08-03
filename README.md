# 🐾 PawPal+ — AI-Assisted Pet Care System

## Original Project (Modules 1–3): PawPal+

My original project was **PawPal+**, a Streamlit app built across Modules 1–3 that helps a busy pet owner stay consistent with pet care. It let an owner enter their info and pets, add care tasks (walks, feeding, meds, grooming) with a time, priority, urgency, duration, and recurrence, and then generated a daily plan. Its core capabilities were a **rule-based scheduler** (`pawpal_system.py`) that sorted tasks by priority/time, filtered by pet or status, detected time-slot conflicts, routed vet visits, deferred tasks outside the owner's availability, and auto-generated the next occurrence of recurring tasks — all backed by a pytest suite and local JSON persistence.

---

## Title and Summary

**PawPal+ now adds a Retrieval-Augmented Generation (RAG) pipeline, a deterministic lab-result interpreter, and an autonomous scheduling agent** on top of the original scheduler. An owner can drop in their pet's real veterinary records (PDF or text); the system retrieves the clinically relevant passages, summarizes the visit into a structured report (labs, diagnoses, prescriptions, vaccines, diet, follow-up) with reference links, **interprets each lab value against its reference range**, and then an agent turns that summary into scheduled care tasks — prescription pickup, medication doses, vaccine boosters — and continuously keeps the schedule in sync with the vet's plan.

**Why it matters:** vet records are dense, inconsistently formatted, and easy to misread — lab reports list a value and a range but usually no HIGH/LOW flag, and it's on the owner to notice what's out of range and what to do next. PawPal+ closes the gap between "here are your dog's records" and "here's exactly what to do and when," while keeping a human in the loop for every clinical decision.

---

## Architecture Overview

The system diagram ([diagrams/uml_rag_agent.mmd](diagrams/uml_rag_agent.mmd), rendered as [Post_Ai_RAG_addition_UML_Final.png](diagrams/Post_Ai_RAG_addition_UML_Final.png)) shows six stages, flowing input → process → output:

1. **Input** — vet notes / lab reports (PDF or text upload) plus owner/pet/task data from the Streamlit session.
2. **Retriever (RAG)** — `chunk_notes()` splits the notes into passages, a pure-Python TF-IDF `NotesIndex` retrieves the top chunks for each clinical aspect (labs, diagnoses, meds, diet, vaccines), and `build_context()` assembles a compact, relevant context.
3. **AI Summarizer (Agent/LLM)** — Claude (`claude-opus-5`) turns that context into a structured `VisitSummary` via a strict JSON schema. When no API key is present, a deterministic offline extractor produces the same structure so the app always works.
4. **Evaluator (deterministic)** — `lab_interpreter.py` compares each lab value to its reference range, assigns a flag and a plain-language clinical note, produces a panel summary, and derives contraindications. This step is rule-based, not model-based, so it is exact and testable.
5. **Vet Care Agent (agentic)** — `build_plan()` converts the summary into vet-managed tasks; `sync()` runs on every schedule change to restore deleted vet tasks and correct any that drifted off the vet's plan, guarded so it **never touches owner-created tasks** and never schedules PRN ("as needed") meds as recurring doses.
6. **Output** — the Scheduler's daily plan with conflicts, the interpreted lab table with summary and ordering links, and an audit log recording every agent action and its reason.

**Human and testing checkpoints** are explicit in the diagram: the owner **approves the care plan** before the agent acts and **reviews the audit log** (edits feed back into `sync`), and the pytest suite has `verifies` edges into the retriever, evaluator, and agent — the three AI-output surfaces.

---

## Setup Instructions

**Prerequisites:** Python 3.11+.

```bash
# 1. Clone and enter the project
cd applied-pethelper-ai-system-project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt   # streamlit, pytest, anthropic, pypdf

# 4. (Optional) Enable the Claude-backed RAG path
export ANTHROPIC_API_KEY=sk-ant-...     # Windows: setx ANTHROPIC_API_KEY sk-ant-...
# Without a key, the app runs the deterministic offline extractor automatically.

# 5. Run the app
streamlit run app.py

# 6. Or run the CLI demo (scheduler + sorting + conflicts)
python main.py

# 7. Run the tests
python -m pytest tests/ -v
```

In the app: fill in Owner Info → add pets → (optionally) add tasks → in **🩺 Vet Records** upload a PDF/paste notes and click **Analyze vet notes** → in **🤖 Vet Care Agent** click **Apply care plan to schedule** → **Generate schedule**.

---

## Sample Interactions

These use real veterinary records for **Montgomery**, a Pembroke Welsh Corgi (VCA/Antech reports), run through the offline path.

### Example 1 — Lab result interpretation & summarization

**Input** (excerpt of a chemistry/CBC/urinalysis panel, no HIGH/LOW flags in the source):

```text
ALT (SGPT) 35 12 - 118 IU/L
BUN 18 6 - 31 mg/dL
SDMA 7.8 <14 ug/dL
Specific Gravity 1.044 1.015 - 1.05
Protein 1+ Negative
Bilirubin 1+ NEG TO 1+
Heartworm No Antigen Detected
```

**AI/Evaluator output:**

```text
39 of 40 interpreted results within normal limits. Out of range: Protein 1+
(abnormal, ref Negative). Protein: proteinuria — protein in the urine;
microalbuminuria testing / renal follow-up may be warranted.
```
The interpreter correctly reads the bounded `SDMA <14` as normal, the qualitative `Bilirubin 1+ (NEG TO 1+)` as normal, the negative heartworm screen, and flags only the trace proteinuria — matching the report's own "Microalbuminuria testing is recommended" note.

### Example 2 — Agent turns the summary into a care plan

**Input:** Montgomery's medication list + vaccine due dates, plus owner availability `["08:00","12:00","18:00"]`.

**Agent output (plan by category):**

```text
pickup:     3   (Credelio, Trazodone, Moxidectin/ProHeart)
medication: 2   (Credelio → 1 monthly dose; ProHeart → daily reminder)
vaccine:    7   (Rabies, Distemper, Parvo, Bordetella, Leptospirosis, Influenza, Lyme,
                 each on its due date — e.g. Rabies due 2028-02-27)
Trazodone scheduled doses: []   ← PRN "as needed for fireworks", so NO recurring doses
```
The agent recognizes that Trazodone is PRN and creates a pickup task but **zero** scheduled doses, while monthly Credelio gets a single monthly dose.

### Example 3 — Scheduler conflict detection (original capability, still intact)

**Input:** two tasks for different pets at the same time (`Morning walk` for Mochi and `Feed Luna` for Luna, both at 08:00).

**Output:**

```text
WARNING Owner conflict at 08:00: 'Feed Luna' (Luna) and 'Morning walk' (Mochi)
overlap - you can't attend both.
```

---

## Design Decisions

- **RAG over a raw prompt dump.** Vet histories can be long and repetitive. Chunking + TF-IDF retrieval keeps the model focused on the clinically relevant passages and controls token cost. I used a **pure-Python TF-IDF index** instead of a vector database — for single-patient documents it's more than accurate enough, and it keeps the project dependency-light and easy to run/grade. *Trade-off:* lexical retrieval can miss pure-synonym matches a semantic embedding would catch; acceptable at this scale.
- **Deterministic interpretation, not model-guessed flags.** Lab classification (value vs. reference range) is done in `lab_interpreter.py`, not by the LLM. The model extracts raw values; the rules decide normal/high/low/abnormal. This makes the most clinically important step **exact, explainable, and unit-testable**, and identical whether or not the API is available. *Trade-off:* the rule set and clinical-note dictionary are hand-curated and finite.
- **Offline fallback everywhere.** If `anthropic` isn't installed or no key is set, `summarize_visit()` falls back to a heuristic extractor. The app never hard-fails on a missing key or network. *Trade-off:* offline extraction of messy free-text (e.g. drug names like "Credelio K9") is approximate; the Claude path cleans it up.
- **Agent guardrails first.** The agent only creates/edits tasks it owns (`source == "agent"`, a vet category) and logs every action with a reason. Owner-created tasks are never modified, and PRN meds are never scheduled as recurring doses. *Trade-off:* the agent is deliberately conservative — it repairs its own plan but won't reorganize the owner's day.
- **Structured output via strict JSON schema.** The LLM call uses `output_config.format` so the response is guaranteed-parseable, plus server-side refusal fallback so a safety decline is retried automatically.
- **Model choice.** Defaulted to `claude-opus-5` for the strongest extraction/reasoning on messy medical text.

---

## Testing Summary

The suite has **134 tests** (`python -m pytest tests/ -v`), all passing in ~1.2s.

| Area | Tests | Coverage |
| --- | --- | --- |
| Original scheduler | 41 | sorting, filtering, recurrence (leap years, rollovers), conflict detection, edge cases |
| RAG + agent | ~50 | retrieval, offline extraction, link builders, frequency parsing, plan generation, `sync` guardrails |
| Lab interpreter + real records | ~43 | range/value parsing, numeric & qualitative classification, tabular + free-text rows, screenings, PRN behavior, real VCA/Antech format |

**What worked:** The deterministic lab interpreter was reliable to test and generalized cleanly from synthetic notes to the real tabular VCA/Antech format — it correctly flagged only the proteinuria across a 40-value panel. The agent's `sync` is idempotent (verified by re-running and asserting no further actions), which made "continuous monitoring" safe.

**What didn't (at first):** Feeding real records surfaced parsing bugs my synthetic tests missed. A medication instruction line ("give every 8-12 hours") was misread as a lab because `8-12` looks like a reference range, and the single-letter `L`/`H` flag matcher caught the `L` in units like `IU/L`. I added instruction-line exclusion, an analyte-name length guard, and a whitespace requirement before flag letters — each pinned down by a new regression test.

**What I learned:** Real-world inputs break assumptions that synthetic fixtures never test, so I added a fixture built from actual patient records. And separating *extraction* (fuzzy, model-friendly) from *interpretation* (deterministic, rule-based) made the system both more trustworthy and far easier to test — the clinically important logic never depends on the model guessing correctly. The main remaining gap is the Streamlit UI layer, which isn't covered by automated tests and would need browser-level testing.
