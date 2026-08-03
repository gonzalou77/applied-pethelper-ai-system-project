from datetime import date
import streamlit as st
from pawpal_system import Pet, Task, Owner, Scheduler, save_to_json, load_from_json
from vet_rag import summarize_visit, claude_available, extract_pdf_text
from vet_agent import VetCareAgent, save_plan, load_plan
from lab_interpreter import interpret_lab_panel

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

st.title("🐾 PawPal+")
st.caption("A pet care planning assistant that builds your pet's daily schedule.")

st.divider()

# --- Owner Setup ---
st.subheader("Owner Info")
col1, col2, col3 = st.columns(3)
with col1:
    owner_name = st.text_input("Owner name", value="Jordan")
with col2:
    owner_age = st.number_input("Age", min_value=1, max_value=120, value=30)
with col3:
    num_pets = st.number_input("Number of pets", min_value=1, max_value=10, value=2)

availability = st.multiselect(
    "Available time slots",
    options=["07:00", "08:00", "09:00", "12:00", "15:00", "18:00", "20:00"],
    default=["08:00", "12:00", "18:00"],
)

st.divider()

# --- Pet Setup ---
st.subheader("Pets")

if "pets" not in st.session_state:
    _saved_pets, _saved_tasks = load_from_json()
    st.session_state.pets = _saved_pets
    st.session_state.tasks = _saved_tasks
    _saved_plan, _saved_audit = load_plan()
    st.session_state.vet_plan = _saved_plan
    st.session_state.vet_audit = _saved_audit

with st.form("add_pet_form", clear_on_submit=True):
    st.markdown("**Add a pet**")
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        pet_name = st.text_input("Pet name", value="Mochi")
    with pcol2:
        pet_species = st.selectbox("Species", ["Dog", "Cat"])
    with pcol3:
        pet_breed = st.text_input("Breed", value="Shiba Inu")

    pcol4, pcol5 = st.columns(2)
    with pcol4:
        pet_dob = st.date_input("Date of birth", value=date(2020, 1, 1))
    with pcol5:
        pet_gotcha = st.date_input("Gotcha day", value=date(2020, 3, 1))

    next_vet = st.date_input("Next vet visit (optional)", value=None)

    if st.form_submit_button("Add pet"):
        if any(p["name"] == pet_name for p in st.session_state.pets):
            st.warning(f"A pet named '{pet_name}' already exists.")
        else:
            st.session_state.pets.append({
                "name": pet_name,
                "species": pet_species,
                "breed": pet_breed,
                "date_of_birth": pet_dob,
                "gotcha_day": pet_gotcha,
                "next_vet_visit": next_vet,
            })
            save_to_json(st.session_state.pets, st.session_state.tasks)
            st.success(f"{pet_name} added!")

if st.session_state.pets:
    st.write("**Your pets:**")
    st.table([{k: str(v) for k, v in p.items()} for p in st.session_state.pets])

    # Build Pet objects from session state and call pawpal_system methods
    live_pets = [
        Pet(
            species=p["species"],
            name=p["name"],
            date_of_birth=p["date_of_birth"],
            breed=p["breed"],
            gotcha_day=p["gotcha_day"],
            next_vet_visit=p.get("next_vet_visit"),
        )
        for p in st.session_state.pets
    ]
    live_owner = Owner(
        name=owner_name,
        number_of_pets=int(num_pets),
        age=int(owner_age),
        availability=availability,
        pets=live_pets,
    )

    with st.expander("Pet Info Summary", expanded=False):
        st.text(live_owner.check_pet_info())

    with st.expander("Daily Care Reminders", expanded=False):
        st.text(live_owner.daily_schedule_check())

    with st.expander("Upcoming Vet Visits", expanded=False):
        st.text(live_owner.vet_visit_schedule())
else:
    st.info("No pets added yet. Use the form above to add your first pet.")

st.divider()

# --- Task Setup ---
st.subheader("Tasks")

if "tasks" not in st.session_state:
    st.session_state.tasks = []  # fallback if page loaded without pets block running

pet_names = [p["name"] for p in st.session_state.pets]

with st.form("add_task_form", clear_on_submit=True):
    st.markdown("**Add a task**")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        task_title = st.text_input("Task title", value="Morning walk")
    with tcol2:
        task_pet = st.selectbox("For which pet?", options=pet_names if pet_names else ["Add a pet first"])

    tcol3, tcol4, tcol5 = st.columns(3)
    with tcol3:
        task_time = st.selectbox("Time", ["07:00", "08:00", "09:00", "12:00", "15:00", "18:00", "20:00"])
    with tcol4:
        task_priority = st.selectbox("Priority", ["low", "medium", "high"], index=2)
    with tcol5:
        task_urgency = st.selectbox("Urgency", ["low", "medium", "high"], index=1)

    tcol6, tcol7 = st.columns(2)
    with tcol6:
        task_duration = st.number_input("Duration (min)", min_value=1, max_value=240, value=30)
    with tcol7:
        task_frequency = st.selectbox("Frequency", ["daily", "weekly", "monthly"])

    if st.form_submit_button("Add task") and pet_names:
        if any(t["title"] == task_title and t["pet"] == task_pet for t in st.session_state.tasks):
            st.warning(f"'{task_title}' already exists for {task_pet}.")
        else:
            st.session_state.tasks.append({
                "title": task_title,
                "pet": task_pet,
                "time": task_time,
                "priority": task_priority,
                "urgency": task_urgency,
                "duration": int(task_duration),
                "frequency": task_frequency,
                "status": "pending",
            })
            save_to_json(st.session_state.pets, st.session_state.tasks)
            st.success(f"Task '{task_title}' added for {task_pet}!")

if st.session_state.tasks:
    st.write("**Current tasks:**")
    st.table(st.session_state.tasks)

    with st.expander("Feeding Schedule (all pets)", expanded=False):
        if st.session_state.pets:
            live_pets = [
                Pet(
                    species=p["species"],
                    name=p["name"],
                    date_of_birth=p["date_of_birth"],
                    breed=p["breed"],
                    gotcha_day=p["gotcha_day"],
                    next_vet_visit=p.get("next_vet_visit"),
                )
                for p in st.session_state.pets
            ]
            live_owner = Owner(
                name=owner_name,
                number_of_pets=int(num_pets),
                age=int(owner_age),
                availability=availability,
                pets=live_pets,
            )
            st.text(live_owner.check_feeding_schedule())
        else:
            st.info("Add pets first to see the feeding schedule.")
else:
    st.info("No tasks yet. Add a task above to get started.")

st.divider()

# --- Vet Records (RAG) ---
st.subheader("🩺 Vet Records & AI Summary")
st.caption(
    "Paste veterinary notes/history below. PawPal+ retrieves the relevant sections "
    "and produces a structured summary: labs, contraindications, diagnoses with "
    "reference links, food and prescription ordering links."
    + ("" if claude_available() else
       " (Claude API credentials not detected — using the offline extractor.)")
)

if "vet_plan" not in st.session_state:
    st.session_state.vet_plan, st.session_state.vet_audit = load_plan()
if "vet_summary" not in st.session_state:
    st.session_state.vet_summary = None

vet_pet = st.selectbox(
    "Which pet are these notes for?",
    options=pet_names if pet_names else ["Add a pet first"],
    key="vet_notes_pet",
)
uploaded = st.file_uploader("Upload lab reports / records (PDF or text)",
                            type=["pdf", "txt"], accept_multiple_files=True)
extracted = ""
if uploaded:
    parts = []
    for f in uploaded:
        if f.name.lower().endswith(".pdf"):
            text = extract_pdf_text(f)
            if not text:
                st.warning(f"Couldn't read text from {f.name}. Install `pypdf`, or paste the text below.")
            parts.append(text)
        else:
            parts.append(f.read().decode("utf-8", errors="ignore"))
    extracted = "\n\n".join(p for p in parts if p)
    if extracted:
        st.caption(f"Extracted {len(extracted)} characters from {len(uploaded)} file(s).")

vet_notes = st.text_area("Veterinary notes / visit history", value=extracted, height=220,
                         placeholder="Paste the vet's visit notes here, or upload files above...")

if st.button("Analyze vet notes", type="primary", disabled=not pet_names):
    if not vet_notes.strip():
        st.warning("Paste or upload some vet notes first.")
    else:
        with st.spinner("Summarizing visit notes..."):
            st.session_state.vet_summary = summarize_visit(vet_notes, pet_name=vet_pet)
        st.success(
            f"Notes analyzed ({'Claude' if st.session_state.vet_summary.generated_by == 'claude' else 'offline extractor'})."
        )

vs = st.session_state.vet_summary
if vs is not None:
    st.markdown("### Visit Summary")
    if vs.visit_date:
        st.caption(f"Visit date: {vs.visit_date}")
    st.write(vs.summary or "—")

    if vs.diagnoses:
        st.markdown("**Diagnosed conditions**")
        for cond in vs.diagnoses:
            st.markdown(
                f"- **{cond.name}** — {cond.summary or 'see references'} "
                f"([Wikipedia]({cond.wiki_url}) · [Merck Vet Manual]({cond.reference_url}))"
            )

    if vs.lab_results:
        st.markdown("**Laboratory results & interpretation**")
        if vs.lab_summary:
            st.info(vs.lab_summary)
        _flag_icon = {"normal": "✅", "high": "🔺", "low": "🔻", "abnormal": "⚠️", "unknown": "❔"}
        st.table([
            {"": _flag_icon.get(l.flag, ""), "Test": l.name, "Value": l.value,
             "Unit": l.unit, "Reference": l.reference_range,
             "Interpretation": l.flag.upper() + (f" — {l.note}" if l.note else "")}
            for l in vs.lab_results
        ])
        abnormal = [l for l in vs.lab_results if l.flag in ("high", "low", "abnormal")]
        if not abnormal:
            st.success("All interpreted lab values are within their reference ranges.")

    if vs.contraindications:
        st.markdown("**Contraindications**")
        for c in vs.contraindications:
            st.warning(f"**{c.finding}** → avoid {c.avoid}. {c.reason}")

    if vs.foods:
        st.markdown("**Recommended food formulations**")
        for f in vs.foods:
            links = " · ".join(f"[{store}]({url})" for store, url in f.order_links.items())
            st.markdown(f"- **{f.product}** — {f.reason} ({links})")

    if vs.prescriptions:
        st.markdown("**Prescriptions**")
        st.table([
            {"Medication": p.name, "Dose": p.dose, "Frequency": p.frequency,
             "Duration": p.duration, "Est. price": p.price_estimate or "—",
             "Refills": p.refills or "—",
             "Pickup needed": "Yes" if p.pickup_required else "No"}
            for p in vs.prescriptions
        ])
        for p in vs.prescriptions:
            links = " · ".join(f"[{store}]({url})" for store, url in p.order_links.items())
            st.markdown(f"- Order **{p.name}**: {links}")

    if vs.vaccines:
        st.markdown("**Vaccines**")
        st.table([
            {"Vaccine": v.name, "Given": v.date_given or "—", "Next due": v.next_due or "—"}
            for v in vs.vaccines
        ])

    if vs.follow_up:
        st.info(f"Follow-up: {vs.follow_up}")

st.divider()

# --- Vet Care Agent ---
st.subheader("🤖 Vet Care Agent")
st.caption(
    "The agent turns the vet summary into scheduled tasks (prescription pickup, "
    "medication doses, vaccine boosters, diet changes, follow-ups) and monitors the "
    "schedule — vet-managed tasks that are deleted or moved off the vet's plan are "
    "restored automatically. It never touches tasks you created yourself."
)

agent = VetCareAgent(availability=availability)

if vs is not None and st.button("Apply care plan to schedule"):
    new_plan = agent.build_plan(vs, pet_name=vet_pet)
    # merge: replace this pet's old plan entries, keep other pets' plans
    kept = [p for p in st.session_state.vet_plan if p.get("pet") != vet_pet]
    st.session_state.vet_plan = kept + new_plan
    st.success(f"Care plan for {vet_pet} added: {len(new_plan)} vet-managed task(s).")

# Constant monitoring: reconcile on every rerun when a plan exists.
if st.session_state.vet_plan:
    synced_tasks, actions = agent.sync(st.session_state.tasks, st.session_state.vet_plan)
    if actions:
        st.session_state.tasks = synced_tasks
        st.session_state.vet_audit.extend(a.to_dict() for a in actions)
        save_to_json(st.session_state.pets, st.session_state.tasks)
    save_plan(st.session_state.vet_plan, st.session_state.vet_audit)
    for a in actions:
        st.info(f"Agent {a.action}: **{a.task_title}** — {a.reason}")

if st.session_state.vet_audit:
    with st.expander(f"Agent audit log ({len(st.session_state.vet_audit)} action(s))", expanded=False):
        st.table([
            {"When": a["timestamp"], "Action": a["action"],
             "Task": a["task_title"], "Pet": a["pet"], "Why": a["reason"]}
            for a in reversed(st.session_state.vet_audit[-25:])
        ])

st.divider()

# --- Generate Schedule ---
st.subheader("Generate Schedule")

sort_by_time = st.toggle("Sort by time instead of priority", value=False)

if st.button("Generate schedule", type="primary"):
    if not st.session_state.pets:
        st.error("Add at least one pet before generating a schedule.")
    elif not st.session_state.tasks:
        st.error("Add at least one task before generating a schedule.")
    else:
        # Build Pet objects
        pet_objects: dict[str, Pet] = {}
        for p in st.session_state.pets:
            pet_objects[p["name"]] = Pet(
                species=p["species"],
                name=p["name"],
                date_of_birth=p["date_of_birth"],
                breed=p["breed"],
                gotcha_day=p["gotcha_day"],
                next_vet_visit=p.get("next_vet_visit"),
            )

        # Build Owner
        owner = Owner(
            name=owner_name,
            number_of_pets=int(num_pets),
            age=int(owner_age),
            availability=availability,
            pets=list(pet_objects.values()),
        )

        # Build and add Tasks
        for t in st.session_state.tasks:
            task = Task(
                title=t["title"],
                frequency=t["frequency"],
                time=t["time"],
                priority=t["priority"],
                status=t["status"],
                duration=t["duration"],
                urgency=t["urgency"],
                pet=pet_objects.get(t["pet"]),
                category=t.get("category", "general"),
                source=t.get("source", "owner"),
                dose=t.get("dose", ""),
            )
            owner.add_task(task)

        # Run Scheduler
        scheduler = Scheduler(owner=owner)
        scheduler.generate_schedule()

        if sort_by_time:
            scheduler.sort_by_time()

        # --- Conflict warnings from detect_conflicts() ---
        conflict_warnings = scheduler.detect_conflicts()
        if conflict_warnings:
            st.markdown("### Scheduling Conflicts Detected")
            for warning in conflict_warnings:
                st.warning(warning)

        # --- Today's Tasks ---
        if scheduler.todays_schedule:
            st.success(
                f"Schedule generated! "
                f"{'Sorted by time.' if sort_by_time else 'Sorted by priority then urgency.'}"
            )
            st.markdown("### Today's Tasks")
            st.table([
                {
                    "Time": t.time,
                    "Task": t.title,
                    "Pet": t.pet.name if t.pet else "—",
                    "Priority": t.priority.capitalize(),
                    "Urgency": t.urgency.capitalize(),
                    "Duration (min)": t.duration,
                    "Frequency": t.frequency.capitalize(),
                }
                for t in scheduler.todays_schedule
            ])

        # --- Vet Visits ---
        if scheduler.vet_visits:
            st.markdown("### Vet Visits")
            st.table([
                {
                    "Time": t.time,
                    "Task": t.title,
                    "Pet": t.pet.name if t.pet else "—",
                    "Duration (min)": t.duration,
                }
                for t in scheduler.vet_visits
            ])

        # --- Bucketed conflicts (tasks bumped during generate_schedule) ---
        if scheduler.conflicts:
            st.markdown("### Tasks Bumped Due to Time Slot Collision")
            st.warning(
                f"{len(scheduler.conflicts)} task(s) share a time slot with an already-scheduled task "
                "and were removed from today's schedule. Move them to the suggested slot below."
            )
            st.table([
                {
                    "Conflicting Time": t.time,
                    "Task": t.title,
                    "Pet": t.pet.name if t.pet else "—",
                    "Priority": t.priority.capitalize(),
                    "Suggested Slot": scheduler.next_available_slot(after=t.time) or "No free slot",
                }
                for t in scheduler.conflicts
            ])

        # --- Deferred Tasks ---
        if scheduler.deferred_tasks:
            st.markdown("### Deferred Tasks")
            st.warning(
                f"{len(scheduler.deferred_tasks)} task(s) fall outside your selected availability "
                "and were not scheduled today. Consider moving them to the suggested slot below."
            )
            st.table([
                {
                    "Original Time": t.time,
                    "Task": t.title,
                    "Pet": t.pet.name if t.pet else "—",
                    "Priority": t.priority.capitalize(),
                    "Suggested Slot": scheduler.next_available_slot() or "No free slot",
                }
                for t in scheduler.deferred_tasks
            ])

        if not scheduler.todays_schedule and not scheduler.vet_visits:
            st.warning("No tasks could be scheduled. Check your availability and task times.")
