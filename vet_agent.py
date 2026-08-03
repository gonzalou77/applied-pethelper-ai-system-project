"""
vet_agent.py — agentic workflow for PawPal+ veterinary scheduling.

The VetCareAgent turns a structured VisitSummary (from vet_rag.py) into a
veterinary care plan — prescription pickup, medication administration per
dose/frequency, vaccine boosters, diet transitions, and follow-up visits —
and continuously reconciles the owner's task list against that plan.

Guardrails:
  - The agent only creates or modifies tasks it owns (source == "agent" and a
    vet category). Owner-created tasks (walks, feeding, play) are never touched.
  - Every action is recorded in an audit log with a reason, so the owner can
    see exactly what changed and why.

"Constant monitoring" model: `sync()` is idempotent and cheap, so callers run
it on every schedule change (in Streamlit: every rerun). Any drift — a deleted
medication task, an administration time moved off the vet plan — is repaired
and logged on the next pass.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Optional

from vet_rag import VisitSummary

# Task categories the agent is allowed to manage. Anything else is owner turf.
VET_CATEGORIES = {"medication", "pickup", "vaccine", "vet_visit", "diet"}

# Fields the agent enforces against the vet plan on its own tasks.
_ENFORCED_FIELDS = ("time", "frequency", "priority", "urgency", "duration", "dose")

PLAN_PATH = "vet_plan.json"


@dataclass
class AgentAction:
    timestamp: str
    action: str        # "created" | "restored" | "corrected" | "skipped"
    task_title: str
    pet: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Frequency interpretation
# ---------------------------------------------------------------------------

_DEFAULT_DOSE_TIMES = ["08:00", "12:00", "18:00", "20:00"]


def parse_frequency(freq_text: str) -> tuple[str, int]:
    """Map a clinical frequency phrase to (app_frequency, doses_per_day).

    app_frequency is one of the PawPal+ Task frequencies: daily/weekly/monthly.
    Examples: "BID" -> ("daily", 2); "q8h" -> ("daily", 3);
              "once weekly" -> ("weekly", 1); "monthly" -> ("monthly", 1).
    """
    t = (freq_text or "").lower()
    if "as needed" in t or "prn" in t:
        return "as_needed", 0   # on-demand — no scheduled administration tasks
    if "month" in t:
        return "monthly", 1
    if "week" in t or "every 7 days" in t:
        return "weekly", 1
    m = re.search(r"q(\d+)h", t)
    if m:
        hours = int(m.group(1))
        return "daily", max(1, min(4, round(24 / hours)))
    if "qid" in t or "four times" in t or "4 times" in t:
        return "daily", 4
    if "tid" in t or "three times" in t or "3 times" in t:
        return "daily", 3
    if "bid" in t or "twice" in t or "2 times" in t:
        return "daily", 2
    # SID, "once daily", "daily", "every other day", PRN, unknown -> once daily
    return "daily", 1


def _pick_dose_times(doses_per_day: int, availability: Optional[list[str]]) -> list[str]:
    """Choose administration time slots, preferring the owner's availability."""
    slots = sorted(availability) if availability else list(_DEFAULT_DOSE_TIMES)
    if not slots:
        slots = list(_DEFAULT_DOSE_TIMES)
    if doses_per_day >= len(slots):
        return slots[:max(1, doses_per_day)] if doses_per_day <= len(slots) else slots
    if doses_per_day == 1:
        return [slots[0]]
    # spread doses across the day: first, last, then evenly between
    picked = [slots[0], slots[-1]]
    step = len(slots) / (doses_per_day - 1) if doses_per_day > 1 else 1
    while len(picked) < doses_per_day:
        idx = round(step * (len(picked) - 1))
        candidate = slots[min(idx, len(slots) - 1)]
        if candidate not in picked:
            picked.append(candidate)
        else:
            for s in slots:
                if s not in picked:
                    picked.append(s)
                    break
            else:
                break
    return sorted(picked)[:doses_per_day]


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class VetCareAgent:
    """Builds and enforces the veterinary portion of a pet's schedule."""

    def __init__(self, availability: Optional[list[str]] = None):
        self.availability = availability or []

    # -- plan generation ----------------------------------------------------

    def build_plan(self, summary: VisitSummary, pet_name: str,
                   today: Optional[date] = None) -> list[dict]:
        """Translate a VisitSummary into vet-managed task dicts.

        Task dicts use the same shape as the app's session/task storage, plus
        `category`, `source`, and `dose` fields that mark them agent-managed.
        """
        today = today or date.today()
        pet = summary.pet_name or pet_name
        plan: list[dict] = []

        def vet_task(title: str, time_slot: str, frequency: str, priority: str,
                     urgency: str, duration: int, category: str, dose: str = "",
                     due_date: str = "") -> dict:
            return {
                "title": title,
                "pet": pet,
                "time": time_slot,
                "priority": priority,
                "urgency": urgency,
                "duration": duration,
                "frequency": frequency,
                "status": "pending",
                "category": category,
                "source": "agent",
                "dose": dose,
                "due_date": due_date or today.isoformat(),
            }

        first_slot = (sorted(self.availability) or _DEFAULT_DOSE_TIMES)[0]

        # 1. Prescription pickup — one task per prescription that needs it
        for rx in summary.prescriptions:
            if rx.pickup_required:
                plan.append(vet_task(
                    title=f"Pick up prescription: {rx.name} for {pet}",
                    time_slot=first_slot, frequency="one-time",
                    priority="high", urgency="high", duration=30,
                    category="pickup", dose=rx.dose,
                ))

        # 2. Medication administration — one task per dose slot.
        #    PRN / as-needed meds (doses_per_day == 0) get no scheduled doses.
        for rx in summary.prescriptions:
            app_freq, doses_per_day = parse_frequency(rx.frequency)
            if doses_per_day == 0:
                continue
            times = _pick_dose_times(doses_per_day, self.availability)
            for i, slot in enumerate(times, start=1):
                dose_label = f" (dose {i} of {len(times)})" if len(times) > 1 else ""
                dose_desc = " ".join(x for x in (rx.dose, rx.frequency) if x)
                plan.append(vet_task(
                    title=f"Give {pet}: {rx.name}{dose_label}",
                    time_slot=slot, frequency=app_freq,
                    priority="high", urgency="high", duration=5,
                    category="medication", dose=dose_desc,
                ))

        # 3. Vaccine boosters — scheduled on their due date when known
        for vac in summary.vaccines:
            if vac.next_due:
                plan.append(vet_task(
                    title=f"Vet visit: {vac.name} vaccine for {pet}",
                    time_slot=first_slot, frequency="one-time",
                    priority="high", urgency="medium", duration=60,
                    category="vaccine", due_date=vac.next_due,
                ))

        # 4. Diet transition — one daily task per recommended formulation
        for food in summary.foods:
            plan.append(vet_task(
                title=f"Feed {pet}: {food.product}",
                time_slot=first_slot, frequency="daily",
                priority="medium", urgency="medium", duration=10,
                category="diet", dose=food.reason,
            ))

        # 5. Follow-up visit
        if summary.follow_up:
            due = self._follow_up_date(summary.follow_up, today)
            plan.append(vet_task(
                title=f"Vet follow-up for {pet}: {summary.follow_up}",
                time_slot=first_slot, frequency="one-time",
                priority="high", urgency="medium", duration=60,
                category="vet_visit", due_date=due.isoformat(),
            ))

        return plan

    @staticmethod
    def _follow_up_date(follow_up_text: str, today: date) -> date:
        """Derive a follow-up date from phrases like 'recheck in 4 weeks'."""
        m = re.search(r"(\d+)\s*(day|week|month)", follow_up_text, re.IGNORECASE)
        if not m:
            return today + timedelta(weeks=2)
        n, unit = int(m.group(1)), m.group(2).lower()
        if unit == "day":
            return today + timedelta(days=n)
        if unit == "week":
            return today + timedelta(weeks=n)
        return today + timedelta(days=30 * n)

    # -- reconciliation / monitoring ----------------------------------------

    @staticmethod
    def _key(task: dict) -> tuple[str, str]:
        return (task.get("title", ""), task.get("pet", ""))

    @staticmethod
    def is_vet_managed(task: dict) -> bool:
        return task.get("source") == "agent" and task.get("category") in VET_CATEGORIES

    def sync(self, tasks: list[dict], plan: list[dict]) -> tuple[list[dict], list[AgentAction]]:
        """Reconcile the live task list against the vet plan.

        Returns (updated task list, actions taken). Idempotent: running it
        twice in a row produces no actions the second time.

        - A plan task missing from the list is (re)created.
        - An agent task whose enforced fields drifted from the plan is reset.
        - Tasks the agent does not own are never modified or removed.
        - Completed ("done") agent tasks are left alone — recurrence is the
          scheduler's job, and re-adding a finished dose would double-dose.
        """
        actions: list[AgentAction] = []
        now = datetime.now().isoformat(timespec="seconds")
        by_key = {self._key(t): t for t in tasks}
        updated = list(tasks)

        for planned in plan:
            key = self._key(planned)
            existing = by_key.get(key)

            if existing is None:
                new_task = dict(planned)
                updated.append(new_task)
                by_key[key] = new_task
                actions.append(AgentAction(
                    timestamp=now, action="created", task_title=planned["title"],
                    pet=planned["pet"],
                    reason="Required by the veterinary care plan and missing from the schedule.",
                ))
                continue

            if not self.is_vet_managed(existing):
                # An owner task happens to share the title — do not touch it.
                actions.append(AgentAction(
                    timestamp=now, action="skipped", task_title=planned["title"],
                    pet=planned["pet"],
                    reason="A task with this name exists but is owner-managed; the agent will not modify it.",
                ))
                continue

            if existing.get("status") == "done":
                continue

            drifted = [f for f in _ENFORCED_FIELDS
                       if f in planned and existing.get(f) != planned.get(f)]
            if drifted:
                for f in drifted:
                    existing[f] = planned[f]
                actions.append(AgentAction(
                    timestamp=now, action="corrected", task_title=planned["title"],
                    pet=planned["pet"],
                    reason=("Reset " + ", ".join(drifted)
                            + " to match the veterinary plan (vaccines/medication "
                              "schedules follow the vet's instructions)."),
                ))

        return updated, actions


# ---------------------------------------------------------------------------
# Plan persistence — keeps the vet plan and audit log across app restarts
# ---------------------------------------------------------------------------


def save_plan(plan: list[dict], audit: list[dict], path: str = PLAN_PATH) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"plan": plan, "audit": audit}, fh, indent=2)


def load_plan(path: str = PLAN_PATH) -> tuple[list[dict], list[dict]]:
    if not os.path.exists(path):
        return [], []
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("plan", []), payload.get("audit", [])
