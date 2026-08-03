"""Tests for the RAG vet-notes pipeline (vet_rag) and the care agent (vet_agent)."""

from datetime import date

import pytest

from vet_rag import (
    NotesIndex,
    VisitSummary,
    Prescription,
    VaccineRecord,
    FoodRecommendation,
    chunk_notes,
    build_context,
    summarize_visit,
    wiki_link,
    food_order_links,
    pharmacy_order_links,
)
from vet_agent import VetCareAgent, parse_frequency


SAMPLE_NOTES = """\
Patient: Mochi (canine, Shiba Inu, 7y)
Visit date: 2026-07-20
Reason: annual exam + owner reports increased thirst

Labs:
ALT 210 U/L (ref 10-125) HIGH
BUN 34 mg/dL (ref 7-27) HIGH
Creatinine 1.4 mg/dL (ref 0.5-1.8)
Glucose 98 mg/dL (ref 74-143)

Assessment:
Early chronic kidney disease suspected. Mild hepatopathy.

Plan:
Start Denamarin 225 mg once daily with food.
Amoxicillin 250 mg twice daily for 10 days.
Transition to Hill's Prescription Diet k/d over 7 days.
Rabies booster due: 2026-09-15
Recheck in 4 weeks with repeat chemistry panel.
"""


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_chunk_notes_splits_on_blank_lines(self):
        chunks = chunk_notes(SAMPLE_NOTES)
        assert len(chunks) >= 3
        assert any("Labs:" in c for c in chunks)
        assert any("Plan:" in c for c in chunks)

    def test_index_retrieves_relevant_chunk_first(self):
        index = NotesIndex(chunk_notes(SAMPLE_NOTES))
        top = index.retrieve("laboratory bloodwork ALT BUN creatinine", k=1)
        assert top and "ALT" in top[0]

    def test_retrieve_on_empty_index(self):
        assert NotesIndex([]).retrieve("anything") == []

    def test_short_notes_pass_through_whole(self):
        assert build_context("short note") == "short note"

    def test_long_notes_get_reduced(self):
        long_notes = "\n\n".join(
            SAMPLE_NOTES for _ in range(20)
        )
        context = build_context(long_notes, max_context_chars=2000)
        assert len(context) <= 2000
        assert context  # something relevant was retrieved


# ---------------------------------------------------------------------------
# Offline extraction (summarize_visit falls back when no API key is set)
# ---------------------------------------------------------------------------

@pytest.fixture
def offline_summary(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    return summarize_visit(SAMPLE_NOTES, pet_name="Mochi")


class TestOfflineExtraction:
    def test_marks_generated_by_offline(self, offline_summary):
        assert offline_summary.generated_by == "offline"

    def test_extracts_flagged_labs(self, offline_summary):
        names = {l.name.upper() for l in offline_summary.lab_results}
        assert "ALT" in names and "BUN" in names
        flags = {l.name.upper(): l.flag for l in offline_summary.lab_results}
        assert flags["ALT"] == "high"
        assert flags["BUN"] == "high"

    def test_contraindications_follow_from_high_labs(self, offline_summary):
        text = " ".join(c.avoid.lower() for c in offline_summary.contraindications)
        assert "nsaid" in text  # both liver and kidney rules mention NSAIDs

    def test_detects_conditions(self, offline_summary):
        names = {d.name for d in offline_summary.diagnoses}
        assert "Chronic kidney disease (CKD)" in names
        assert "Hepatopathy" in names

    def test_condition_links_populated(self, offline_summary):
        for d in offline_summary.diagnoses:
            assert d.wiki_url.startswith("https://en.wikipedia.org/")
            assert d.reference_url.startswith("https://www.merckvetmanual.com/")

    def test_extracts_prescriptions_with_dose_and_frequency(self, offline_summary):
        rx = {p.name.lower(): p for p in offline_summary.prescriptions}
        assert "denamarin" in rx and "amoxicillin" in rx
        assert "225" in rx["denamarin"].dose
        assert "twice" in rx["amoxicillin"].frequency.lower()

    def test_prescription_order_links(self, offline_summary):
        for p in offline_summary.prescriptions:
            assert any("chewy.com" in url for url in p.order_links.values())

    def test_detects_vaccine_with_due_date(self, offline_summary):
        rabies = [v for v in offline_summary.vaccines if "rabies" in v.name.lower()]
        assert rabies and rabies[0].next_due == "2026-09-15"

    def test_detects_food_recommendation(self, offline_summary):
        assert offline_summary.foods
        assert "k/d" in offline_summary.foods[0].product.lower()

    def test_captures_follow_up(self, offline_summary):
        assert "recheck" in offline_summary.follow_up.lower()

    def test_visit_date_extracted(self, offline_summary):
        assert offline_summary.visit_date == "2026-07-20"


# ---------------------------------------------------------------------------
# Link builders
# ---------------------------------------------------------------------------

class TestLinks:
    def test_wiki_link_encodes_spaces(self):
        assert "Chronic+kidney+disease" in wiki_link("Chronic kidney disease")

    def test_food_links_include_stores(self):
        links = food_order_links("Hill's k/d")
        assert "Chewy" in links and "Amazon" in links

    def test_pharmacy_links_include_pharmacies(self):
        links = pharmacy_order_links("Amoxicillin 250 mg")
        assert "Chewy Pharmacy" in links and "PetMeds" in links


# ---------------------------------------------------------------------------
# Frequency parsing
# ---------------------------------------------------------------------------

class TestFrequencyParsing:
    @pytest.mark.parametrize("text,expected", [
        ("once daily", ("daily", 1)),
        ("SID", ("daily", 1)),
        ("twice daily", ("daily", 2)),
        ("BID", ("daily", 2)),
        ("q12h", ("daily", 2)),
        ("three times daily", ("daily", 3)),
        ("q8h", ("daily", 3)),
        ("once weekly", ("weekly", 1)),
        ("monthly", ("monthly", 1)),
        ("", ("daily", 1)),
    ])
    def test_parse_frequency(self, text, expected):
        assert parse_frequency(text) == expected


# ---------------------------------------------------------------------------
# Care agent — plan generation
# ---------------------------------------------------------------------------

def make_summary() -> VisitSummary:
    return VisitSummary(
        pet_name="Mochi",
        prescriptions=[
            Prescription(name="Amoxicillin", dose="250 mg", frequency="twice daily",
                         pickup_required=True),
            Prescription(name="Denamarin", dose="225 mg", frequency="once daily",
                         pickup_required=False),
        ],
        vaccines=[VaccineRecord(name="Rabies", next_due="2026-09-15")],
        foods=[FoodRecommendation(product="Hill's k/d", reason="kidney support")],
        follow_up="Recheck in 4 weeks",
    )


@pytest.fixture
def agent():
    return VetCareAgent(availability=["08:00", "12:00", "18:00"])


@pytest.fixture
def plan(agent):
    return agent.build_plan(make_summary(), pet_name="Mochi", today=date(2026, 8, 1))


class TestPlanGeneration:
    def test_pickup_task_only_when_required(self, plan):
        pickups = [t for t in plan if t["category"] == "pickup"]
        assert len(pickups) == 1
        assert "Amoxicillin" in pickups[0]["title"]

    def test_bid_medication_creates_two_dose_tasks(self, plan):
        doses = [t for t in plan if t["category"] == "medication" and "Amoxicillin" in t["title"]]
        assert len(doses) == 2
        assert {t["time"] for t in doses} == {"08:00", "18:00"}

    def test_once_daily_medication_creates_one_task(self, plan):
        doses = [t for t in plan if t["category"] == "medication" and "Denamarin" in t["title"]]
        assert len(doses) == 1

    def test_vaccine_task_uses_due_date(self, plan):
        vac = [t for t in plan if t["category"] == "vaccine"]
        assert vac and vac[0]["due_date"] == "2026-09-15"

    def test_diet_task_created(self, plan):
        diet = [t for t in plan if t["category"] == "diet"]
        assert diet and diet[0]["frequency"] == "daily"

    def test_follow_up_scheduled_four_weeks_out(self, plan):
        visits = [t for t in plan if t["category"] == "vet_visit"]
        assert visits and visits[0]["due_date"] == "2026-08-29"

    def test_all_plan_tasks_are_agent_sourced(self, plan):
        assert all(t["source"] == "agent" for t in plan)


# ---------------------------------------------------------------------------
# Care agent — sync / monitoring guardrails
# ---------------------------------------------------------------------------

def owner_task(title="Morning walk", pet="Mochi"):
    return {
        "title": title, "pet": pet, "time": "08:00", "priority": "high",
        "urgency": "high", "duration": 30, "frequency": "daily",
        "status": "pending",
    }


class TestAgentSync:
    def test_creates_missing_plan_tasks(self, agent, plan):
        tasks, actions = agent.sync([owner_task()], plan)
        assert len(tasks) == 1 + len(plan)
        assert all(a.action == "created" for a in actions)

    def test_sync_is_idempotent(self, agent, plan):
        tasks, _ = agent.sync([], plan)
        tasks2, actions2 = agent.sync(tasks, plan)
        assert actions2 == []
        assert len(tasks2) == len(tasks)

    def test_never_modifies_owner_tasks(self, agent, plan):
        walk = owner_task()
        original = dict(walk)
        tasks, _ = agent.sync([walk], plan)
        assert walk == original
        assert walk in tasks

    def test_restores_deleted_vet_task(self, agent, plan):
        tasks, _ = agent.sync([], plan)
        # user deletes a medication task
        tasks = [t for t in tasks if "Amoxicillin (dose 1" not in t["title"]]
        tasks, actions = agent.sync(tasks, plan)
        assert any(a.action == "created" for a in actions)
        assert any("Amoxicillin (dose 1" in t["title"] for t in tasks)

    def test_corrects_time_drift_on_vet_task(self, agent, plan):
        tasks, _ = agent.sync([], plan)
        med = next(t for t in tasks if t["category"] == "medication")
        med["time"] = "23:00"  # user moves the dose off the vet plan
        tasks, actions = agent.sync(tasks, plan)
        assert any(a.action == "corrected" for a in actions)
        assert med["time"] != "23:00"

    def test_owner_task_with_same_title_is_skipped_not_modified(self, agent):
        summary = make_summary()
        plan = agent.build_plan(summary, pet_name="Mochi")
        clashing = owner_task(title=plan[0]["title"])
        clashing["time"] = "12:00"
        tasks, actions = agent.sync([clashing], plan)
        assert clashing["time"] == "12:00"  # untouched
        assert any(a.action == "skipped" for a in actions)

    def test_done_vet_tasks_left_alone(self, agent, plan):
        tasks, _ = agent.sync([], plan)
        med = next(t for t in tasks if t["category"] == "medication")
        med["status"] = "done"
        med["time"] = "23:00"  # even drifted, a done task is not corrected
        _, actions = agent.sync(tasks, plan)
        assert all(a.task_title != med["title"] or a.action != "corrected" for a in actions)


# ---------------------------------------------------------------------------
# Lab interpreter — the prototype interpretation/summarization feature
# ---------------------------------------------------------------------------

from lab_interpreter import (
    parse_reference_range,
    parse_numeric_value,
    classify,
    classify_qualitative,
    parse_lab_line,
    interpret_lab_panel,
    interpret_findings,
    detect_screenings,
)


class TestReferenceRangeParsing:
    @pytest.mark.parametrize("ref,expected", [
        ("5 - 7.4", (5.0, 7.4, True)),
        ("6 - 31", (6.0, 31.0, True)),
        ("<14", (None, 14.0, True)),
        ("< 14", (None, 14.0, True)),
        (">2", (2.0, None, True)),
        ("1.015 - 1.05", (1.015, 1.05, True)),
        ("Negative", (None, None, False)),
        ("", (None, None, False)),
    ])
    def test_parse_reference_range(self, ref, expected):
        assert parse_reference_range(ref) == expected

    def test_parse_numeric_value(self):
        assert parse_numeric_value("6.1") == (6.1, 6.1)
        assert parse_numeric_value("0-1") == (0.0, 1.0)
        assert parse_numeric_value("1+") is None
        assert parse_numeric_value("NEGATIVE") is None


class TestClassification:
    @pytest.mark.parametrize("value,ref,flag", [
        ("6.1", "5 - 7.4", "normal"),
        ("35", "12 - 118", "normal"),
        ("210", "12 - 118", "high"),
        ("3", "5 - 7.4", "low"),
        ("7.8", "<14", "normal"),
        ("18", "<14", "high"),
        ("0-1", "0-3", "normal"),
        ("2-3", "0-3", "normal"),
    ])
    def test_numeric_classification(self, value, ref, flag):
        assert classify(value, ref) == flag

    @pytest.mark.parametrize("value,ref,flag", [
        ("NEGATIVE", "Negative", "normal"),
        ("1+", "Negative", "abnormal"),
        ("1+", "NEG TO 1+", "normal"),
        ("2+", "NEG TO 1+", "abnormal"),
        ("NONE SEEN", "None seen", "normal"),
    ])
    def test_qualitative_classification(self, value, ref, flag):
        assert classify_qualitative(value, ref) == flag


class TestLabLineParsing:
    def test_parses_core_chem_row(self):
        f = parse_lab_line("ALT (SGPT) 35 12 - 118 IU/L")
        assert f.name == "ALT (SGPT)"
        assert f.value == "35"
        assert f.unit == "IU/L"
        assert f.reference_range == "12 - 118"
        assert f.flag == "normal"

    def test_parses_bounded_ref(self):
        f = parse_lab_line("SDMA 7.8 <14 ug/dL")
        assert f.reference_range == "<14" and f.flag == "normal"

    def test_ratio_row_without_unit(self):
        f = parse_lab_line("A/G RATIO 1.0 0.8 - 2")
        assert f.value == "1.0" and f.unit == "" and f.flag == "normal"

    def test_proteinuria_row_is_abnormal_with_note(self):
        f = parse_lab_line("Protein 1+ Negative")
        assert f.flag == "abnormal"
        assert "proteinuria" in f.note.lower()

    def test_rejects_non_lab_lines(self):
        assert parse_lab_line("Weight: 31 lb") is None
        assert parse_lab_line("Credelio K9 25.1-50lb/11-22kg Chew") is None
        assert parse_lab_line("K9 Rabies 27-Feb-2028") is None
        assert parse_lab_line("5y 6m 14.06 kg 31 lb 20-May-2026") is None

    def test_rejects_hospital_address_footer(self):
        # phone "474 - 2454" looks like a reference range; must not become a lab
        assert parse_lab_line(
            "2917 Old US 231 South, Lafayette, IN 47909 | 765 474 - 2454") is None
        assert parse_lab_line("231 South, Lafayette, IN 47909 | 765") is None
        assert parse_lab_line("474 - 2454") is None

    def test_footer_not_flagged_in_full_panel(self):
        text = (REAL_PANEL
                + "\nVCA Paw Prints Animal Hospital"
                + "\n2917 Old US 231 South, Lafayette, IN 47909 | 765 474 - 2454\n")
        panel = interpret_lab_panel(text)
        assert all("2917" not in f.name and "Old US" not in f.name
                   for f in panel.findings)
        assert len(panel.abnormal()) == 1  # still just the proteinuria

    def test_high_value_gets_clinical_note(self):
        f = parse_lab_line("ALT (SGPT) 210 12 - 118 IU/L")
        assert f.flag == "high" and "liver" in f.note.lower()


class TestScreenings:
    def test_detects_negative_screens(self):
        text = ("Heartworm No Antigen Detected\n"
                "Borrelia burgdorferi Negative\n"
                "GIARDIA (ELISA) NEGATIVE")
        screens = detect_screenings(text)
        names = {s.name for s in screens}
        assert "Heartworm antigen" in names
        assert "Giardia" in names
        assert all(s.flag == "normal" for s in screens)


REAL_PANEL = """\
CoreChem
TOTAL PROTEIN 6.1 5 - 7.4 g/dL
ALT (SGPT) 35 12 - 118 IU/L
BUN 18 6 - 31 mg/dL
CREATININE 0.9 0.5 - 1.6 mg/dL
SDMA 7.8 <14 ug/dL
GLUCOSE 98 70 - 138 mg/dL
Complete Blood Count
HGB 18.9 12.1 - 20.3 g/dL
HCT 58 36 - 60 %
Platelet Count 253 170 - 400 10^3/uL
Urinalysis-Complete
Specific Gravity 1.044 1.015 - 1.05
pH 6.0 5.5 - 7
Protein 1+ Negative
Bilirubin 1+ NEG TO 1+
Accuplex
Heartworm No Antigen Detected
Borrelia burgdorferi Negative
"""


class TestRealPanelInterpretation:
    """Interpretation of the actual VCA/Antech tabular report format."""

    def test_panel_flags_only_proteinuria(self):
        panel = interpret_lab_panel(REAL_PANEL)
        abnormal = panel.abnormal()
        assert len(abnormal) == 1
        assert abnormal[0].name == "Protein"
        assert abnormal[0].flag == "abnormal"

    def test_panel_summary_mentions_within_normal_limits(self):
        panel = interpret_lab_panel(REAL_PANEL)
        assert "within normal limits" in panel.summary.lower()

    def test_bounded_and_qualitative_ranges_read_correctly(self):
        panel = interpret_lab_panel(REAL_PANEL)
        by_name = {f.name: f for f in panel.findings}
        assert by_name["SDMA"].flag == "normal"        # 7.8 < 14
        assert by_name["Bilirubin"].flag == "normal"   # 1+ within NEG TO 1+
        assert by_name["Heartworm antigen"].flag == "normal"

    def test_all_core_chem_within_range(self):
        panel = interpret_lab_panel(REAL_PANEL)
        core = [f for f in panel.findings if f.name in
                ("TOTAL PROTEIN", "ALT (SGPT)", "BUN", "CREATININE", "GLUCOSE")]
        assert core and all(f.flag == "normal" for f in core)


class TestInterpretFindings:
    def test_reclassifies_structured_rows_from_llm(self):
        rows = [
            {"name": "ALT", "value": "210", "reference_range": "12 - 118", "unit": "IU/L"},
            {"name": "BUN", "value": "18", "reference_range": "6 - 31", "unit": "mg/dL"},
        ]
        panel = interpret_findings(rows)
        flags = {f.name: f.flag for f in panel.findings}
        assert flags["ALT"] == "high"   # recomputed even if the model said otherwise
        assert flags["BUN"] == "normal"
        assert "liver" in next(f.note for f in panel.findings if f.name == "ALT").lower()


# ---------------------------------------------------------------------------
# Real-record extraction through summarize_visit (offline path)
# ---------------------------------------------------------------------------

REAL_RECORD = REAL_PANEL + """
Ova & Parasite NONE SEEN
GIARDIA (ELISA) NEGATIVE
Most recent visit date: 20-May-2026
Medications
Credelio K9 25.1-50lb/11-22kg Chew: Give 1 chewable tablet by mouth once a month for flea and tick control.
TraZODone HCL (gen) 100mg Tab: Give 1 tablet by mouth every 8-12 hours as needed for fireworks anxiety.
K9 Rabies 27-Feb-2028
K9 Bordetella 20-Apr-2027
"""


@pytest.fixture
def real_summary(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    return summarize_visit(REAL_RECORD, pet_name="Montgomery")


class TestRealRecordExtraction:
    def test_visit_date_normalized_to_iso(self, real_summary):
        assert real_summary.visit_date == "2026-05-20"

    def test_lab_summary_populated(self, real_summary):
        assert "within normal limits" in real_summary.lab_summary.lower()

    def test_only_proteinuria_flagged(self, real_summary):
        abnormal = [l for l in real_summary.lab_results
                    if l.flag in ("high", "low", "abnormal")]
        assert [l.name for l in abnormal] == ["Protein"]

    def test_monthly_med_extracted(self, real_summary):
        credelio = [p for p in real_summary.prescriptions if "credelio" in p.name.lower()]
        assert credelio and "month" in credelio[0].frequency.lower()

    def test_prn_med_marked_as_needed(self, real_summary):
        traz = [p for p in real_summary.prescriptions if "trazodone" in p.name.lower()]
        assert traz and "as needed" in traz[0].frequency.lower()

    def test_vaccine_dates_normalized(self, real_summary):
        rabies = [v for v in real_summary.vaccines if v.name == "Rabies"]
        assert rabies and rabies[0].next_due == "2028-02-27"


class TestPrnAgentBehavior:
    def test_prn_medication_creates_no_dose_tasks(self, real_summary):
        agent = VetCareAgent(availability=["08:00", "12:00", "18:00"])
        plan = agent.build_plan(real_summary, pet_name="Montgomery")
        traz_doses = [t for t in plan
                      if t["category"] == "medication" and "TraZODone" in t["title"]]
        assert traz_doses == []
        # but it still gets a pickup task
        traz_pickup = [t for t in plan
                       if t["category"] == "pickup" and "TraZODone" in t["title"]]
        assert len(traz_pickup) == 1

    def test_monthly_med_creates_one_monthly_dose(self, real_summary):
        agent = VetCareAgent(availability=["08:00", "12:00", "18:00"])
        plan = agent.build_plan(real_summary, pet_name="Montgomery")
        credelio = [t for t in plan
                    if t["category"] == "medication" and "Credelio" in t["title"]]
        assert len(credelio) == 1 and credelio[0]["frequency"] == "monthly"
