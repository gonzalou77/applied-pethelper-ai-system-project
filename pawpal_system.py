from __future__ import annotations
import calendar
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import combinations
from typing import Optional


PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
URGENCY_ORDER  = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Pet:
    species: str
    name: str
    date_of_birth: date
    breed: str
    gotcha_day: date
    next_vet_visit: Optional[date] = None
    flea_tick_due: Optional[date] = None
    heartworm_due: Optional[date] = None

    def check_feeding_schedule(self) -> str:
        """Return a feeding reminder string based on the pet's species."""
        if self.species.lower() == "dog":
            return f"{self.name} should be fed twice a day: morning and evening."
        return f"{self.name} should be fed twice a day or have food available."

    def water_refill_notification(self) -> str:
        """Return a reminder to refill the pet's water bowl."""
        return f"Refill {self.name}'s water bowl at least twice today."

    def vaccination_schedule_check(self) -> str:
        """Return any overdue flea, tick, or heartworm treatments for this pet."""
        today = date.today()
        lines = []
        if self.flea_tick_due and self.flea_tick_due <= today:
            lines.append(f"{self.name}: flea/tick treatment is due ({self.flea_tick_due}).")
        if self.heartworm_due and self.heartworm_due <= today:
            lines.append(f"{self.name}: heartworm treatment is due ({self.heartworm_due}).")
        return "\n".join(lines) if lines else f"{self.name} is up to date on treatments."

    def daily_schedule_check(self) -> str:
        """Return a combined daily summary of feeding, water, and vaccination checks."""
        checks = [
            self.check_feeding_schedule(),
            self.water_refill_notification(),
            self.vaccination_schedule_check(),
        ]
        return "\n".join(checks)


@dataclass
class Task:
    title: str
    frequency: str       # "daily", "weekly", "monthly"
    time: str            # "08:00" — 24-hour format
    priority: str        # "low", "medium", "high"
    status: str          # "pending", "done"
    duration: int        # minutes
    urgency: str         # "low", "medium", "high"
    owner: Optional[Owner] = field(default=None, repr=False)
    pet: Optional[Pet] = field(default=None, repr=False)
    due_date: date = field(default_factory=date.today)
    category: str = "general"   # "general" | "medication" | "pickup" | "vaccine" | "vet_visit" | "diet"
    source: str = "owner"       # "owner" (user-created) | "agent" (vet care agent)
    dose: str = ""              # e.g. "5 mg twice daily" — populated on medication tasks

    def mark_done(self) -> None:
        """Set the task status to done."""
        self.status = "done"

    def is_pending(self) -> bool:
        """Return True if the task has not been completed yet."""
        return self.status == "pending"

    def next_occurrence(self) -> Task:
        """
        Calculate and return a new pending Task for the next scheduled occurrence.

        Uses timedelta for daily and weekly frequencies. Monthly recurrence uses
        calendar.monthrange to find the true last day of the target month, then
        clamps the day to that value — this correctly handles variable month lengths
        (e.g. Jan 31 -> Feb 28, or Feb 29 in a leap year) and year rollovers
        (e.g. Dec -> Jan of the following year). Unknown frequencies fall back to daily.

        Returns a full copy of this task with status reset to 'pending' and
        due_date advanced by the appropriate interval.
        """
        d = self.due_date
        if self.frequency == "daily":
            next_date = d + timedelta(days=1)
        elif self.frequency == "weekly":
            next_date = d + timedelta(weeks=1)
        elif self.frequency == "monthly":
            month = d.month % 12 + 1
            year = d.year + (1 if d.month == 12 else 0)
            # clamp day to the last valid day of the target month (handles Feb 28/29, 30-day months)
            day = min(d.day, calendar.monthrange(year, month)[1])
            next_date = d.replace(year=year, month=month, day=day)
        else:
            next_date = d + timedelta(days=1)
        return Task(
            title=self.title,
            frequency=self.frequency,
            time=self.time,
            priority=self.priority,
            status="pending",
            duration=self.duration,
            urgency=self.urgency,
            owner=self.owner,
            pet=self.pet,
            due_date=next_date,
            category=self.category,
            source=self.source,
            dose=self.dose,
        )


@dataclass
class Owner:
    name: str
    number_of_pets: int
    age: int
    availability: list[str] = field(default_factory=list)  # e.g. ["08:00", "12:00", "18:00"]
    pets: list[Pet] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)

    def add_task(self, task: Task) -> None:
        """Assign this owner to the task and add it to the owner's task list."""
        task.owner = self
        self.tasks.append(task)

    def remove_task(self, task: Task) -> None:
        """Remove a task from the owner's task list if it exists."""
        if task in self.tasks:
            self.tasks.remove(task)

    def edit_task(self, task: Task, **updates) -> None:
        """Update any attributes on a task by passing keyword arguments."""
        for attr, value in updates.items():
            if hasattr(task, attr):
                setattr(task, attr, value)

    def check_feeding_schedule(self) -> str:
        """Return feeding reminders for all of this owner's pets."""
        return "\n".join(pet.check_feeding_schedule() for pet in self.pets)

    def check_pet_info(self) -> str:
        """Return a summary of basic info for each of the owner's pets."""
        lines = []
        for pet in self.pets:
            lines.append(
                f"{pet.name} | {pet.species} | {pet.breed} | "
                f"Born: {pet.date_of_birth} | Gotcha: {pet.gotcha_day}"
            )
        return "\n".join(lines)

    def vaccination_schedule_check(self) -> str:
        """Return vaccination and treatment status for all of the owner's pets."""
        return "\n".join(pet.vaccination_schedule_check() for pet in self.pets)

    def daily_schedule_check(self) -> str:
        """Return the full daily care summary for all of the owner's pets."""
        return "\n".join(pet.daily_schedule_check() for pet in self.pets)

    def vet_visit_schedule(self) -> str:
        """Return upcoming vet visit dates for all of the owner's pets."""
        lines = []
        for pet in self.pets:
            if pet.next_vet_visit:
                lines.append(f"{pet.name}: next vet visit on {pet.next_vet_visit}.")
        return "\n".join(lines) if lines else "No upcoming vet visits scheduled."

    def all_pet_tasks(self) -> list[Task]:
        """Collect every task linked to any of this owner's pets."""
        pet_tasks = []
        for task in self.tasks:
            if task.pet in self.pets:
                pet_tasks.append(task)
        return pet_tasks


@dataclass
class Scheduler:
    owner: Owner
    todays_schedule: list[Task] = field(default_factory=list)
    deferred_tasks: list[Task] = field(default_factory=list)
    conflicts: list[Task] = field(default_factory=list)
    vet_visits: list[Task] = field(default_factory=list)

    def generate_schedule(self) -> None:
        """
        Pull all pet-linked tasks from the owner, then bucket them:
        - vet visits go to vet_visits
        - tasks whose time falls outside owner availability go to deferred_tasks
        - overlapping tasks (same time slot) go to conflicts
        - everything else goes to todays_schedule
        """
        self.todays_schedule.clear()
        self.deferred_tasks.clear()
        self.conflicts.clear()
        self.vet_visits.clear()

        pending_tasks = [t for t in self.owner.all_pet_tasks() if t.is_pending()]

        seen_times: dict[str, Task] = {}

        for task in pending_tasks:
            if "vet" in task.title.lower():
                self.vet_visits.append(task)
                continue

            if self.owner.availability and task.time not in self.owner.availability:
                self.deferred_tasks.append(task)
                continue

            if task.time in seen_times:
                self.conflicts.append(task)
            else:
                seen_times[task.time] = task
                self.todays_schedule.append(task)

        self.sort_schedule()

    def sort_schedule(self) -> None:
        """Sort todays_schedule by priority then time."""
        self.todays_schedule.sort(key=lambda t: (
            PRIORITY_ORDER.get(t.priority, 99),
            t.time,
        ))

    def filter_tasks(self, status: Optional[str] = None, pet_name: Optional[str] = None) -> list[Task]:
        """
        Return a filtered subset of todays_schedule matching the given criteria.

        Both parameters are optional and act as AND conditions when combined:
          - status:   match tasks whose status equals this value (e.g. "pending", "done")
          - pet_name: match tasks assigned to a pet with this exact name

        Passing neither argument returns all tasks in todays_schedule unchanged.
        Tasks with no assigned pet are excluded when pet_name is provided.
        Does not mutate todays_schedule — always returns a new list.
        """
        return [
            t for t in self.todays_schedule
            if (status is None or t.status == status)
            and (pet_name is None or (t.pet is not None and t.pet.name == pet_name))
        ]

    def sort_by_time(self) -> None:
        """
        Sort todays_schedule into chronological order using each task's time string.

        Relies on the fact that zero-padded HH:MM strings sort correctly as plain
        strings (lexicographic order matches time order), so no parsing is needed.
        Mutates todays_schedule in place. Call after generate_schedule() to override
        the default priority-first ordering with a time-first view.
        """
        self.todays_schedule.sort(key=lambda t: t.time)

    def detect_conflicts(self) -> list[str]:
        """
        Scan all pending pet tasks and return human-readable warning messages for
        any time slot collisions. Never raises — returns an empty list if no conflicts.

        Algorithm:
          1. Collect all pending tasks across every pet the owner owns.
          2. Group them into buckets by time slot using a defaultdict.
          3. For each bucket with two or more tasks, generate every unique pair
             using itertools.combinations (avoids duplicate (a,b)/(b,a) comparisons).
          4. Classify each pair:
               - Same pet name -> "Same-pet conflict" (pet double-booked)
               - Different pet names -> "Owner conflict" (owner can't be in two places)

        Returns a list of warning strings, one per conflicting pair. An empty list
        means the schedule is clean.
        """
        pending = [t for t in self.owner.all_pet_tasks() if t.is_pending()]

        by_time: defaultdict[str, list[Task]] = defaultdict(list)
        for task in pending:
            by_time[task.time].append(task)

        warnings: list[str] = []
        for time_slot, tasks in by_time.items():
            for a, b in combinations(tasks, 2):
                pet_a = a.pet.name if a.pet else "Unknown"
                pet_b = b.pet.name if b.pet else "Unknown"
                if pet_a == pet_b:
                    warnings.append(
                        f"WARNING Same-pet conflict at {time_slot}: '{a.title}' and '{b.title}' "
                        f"are both scheduled for {pet_a}."
                    )
                else:
                    warnings.append(
                        f"WARNING Owner conflict at {time_slot}: '{a.title}' ({pet_a}) and "
                        f"'{b.title}' ({pet_b}) overlap - you can't attend both."
                    )
        return warnings

    def complete_task(self, task: Task) -> Optional[Task]:
        """Mark a task done and, if it recurs, register the next occurrence with the owner."""
        task.mark_done()
        if task.frequency in ("daily", "weekly", "monthly"):
            next_task = task.next_occurrence()
            if self.owner:
                self.owner.add_task(next_task)
            return next_task
        return None

    def next_available_slot(self, after: Optional[str] = None) -> Optional[str]:
        """
        Return the first time slot in owner.availability that is not already
        occupied in todays_schedule, or None if every slot is taken.

        Parameters
        ----------
        after : str, optional
            A zero-padded HH:MM string. When supplied, only slots that sort
            strictly after this value are considered. This lets callers find
            a replacement slot for a specific conflicted or deferred task
            (e.g. pass the task's own time to skip it and look forward).

        Algorithm
        ---------
        1. Collect the set of times already claimed by todays_schedule.
        2. Sort owner.availability lexicographically (HH:MM strings sort
           correctly without parsing).
        3. Walk the sorted list and return the first slot that is both free
           and, when `after` is given, strictly greater than `after`.
        """
        occupied = {t.time for t in self.todays_schedule}
        for slot in sorted(self.owner.availability):
            if after is not None and slot <= after:
                continue
            if slot not in occupied:
                return slot
        return None

    def edit_schedule(self, task: Task, **updates) -> None:
        """Edit a task's attributes and regenerate the schedule."""
        self.owner.edit_task(task, **updates)
        self.generate_schedule()

    def summary(self) -> str:
        """Return a formatted string of today's schedule, vet visits, conflicts, and deferred tasks."""
        lines = [f"Schedule for {self.owner.name}'s pets\n{'='*40}"]
        if self.todays_schedule:
            lines.append("\nToday's Tasks:")
            for t in self.todays_schedule:
                lines.append(f"  [{t.time}] {t.title} ({t.duration} min) — {t.priority} priority")
        if self.vet_visits:
            lines.append("\nVet Visits:")
            for t in self.vet_visits:
                lines.append(f"  [{t.time}] {t.title}")
        if self.conflicts:
            lines.append("\nConflicts (same time slot):")
            for t in self.conflicts:
                lines.append(f"  [{t.time}] {t.title} — needs rescheduling")
        if self.deferred_tasks:
            lines.append("\nDeferred (outside availability):")
            for t in self.deferred_tasks:
                lines.append(f"  [{t.time}] {t.title}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON persistence helpers
# ---------------------------------------------------------------------------

_DATE_FIELDS_PET  = ("date_of_birth", "gotcha_day", "next_vet_visit")
_DATE_FIELDS_TASK = ("due_date",)


def _serialize_pet(pet: dict) -> dict:
    """Convert a pet dict's date values to ISO strings for JSON serialisation."""
    out = dict(pet)
    for key in _DATE_FIELDS_PET:
        val = out.get(key)
        out[key] = val.isoformat() if isinstance(val, date) else val
    return out


def _deserialize_pet(pet: dict) -> dict:
    """Convert ISO date strings back to date objects in a pet dict."""
    out = dict(pet)
    for key in _DATE_FIELDS_PET:
        val = out.get(key)
        out[key] = date.fromisoformat(val) if isinstance(val, str) else val
    return out


def _serialize_task(task: dict) -> dict:
    """Convert a task dict's date values to ISO strings for JSON serialisation."""
    out = dict(task)
    for key in _DATE_FIELDS_TASK:
        val = out.get(key)
        out[key] = val.isoformat() if isinstance(val, date) else val
    return out


def _deserialize_task(task: dict) -> dict:
    """Convert ISO date strings back to date objects in a task dict."""
    out = dict(task)
    for key in _DATE_FIELDS_TASK:
        val = out.get(key)
        out[key] = date.fromisoformat(val) if isinstance(val, str) else val
    return out


def save_to_json(pets: list[dict], tasks: list[dict], path: str = "data.json") -> None:
    """
    Persist the current pets and tasks to a JSON file.

    Accepts the raw dict lists that Streamlit stores in session_state.
    date objects are serialised to ISO-8601 strings (YYYY-MM-DD); None
    values are preserved as JSON null.

    Parameters
    ----------
    pets  : list of pet dicts from session_state.pets
    tasks : list of task dicts from session_state.tasks
    path  : destination file path (default: data.json in the working directory)
    """
    payload = {
        "pets":  [_serialize_pet(p)  for p in pets],
        "tasks": [_serialize_task(t) for t in tasks],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_from_json(path: str = "data.json") -> tuple[list[dict], list[dict]]:
    """
    Load pets and tasks from a JSON file previously written by save_to_json.

    Returns (pets, tasks) as lists of dicts ready to be assigned back to
    session_state. ISO date strings are converted back to date objects.
    Returns two empty lists when the file does not exist yet.

    Parameters
    ----------
    path : source file path (default: data.json in the working directory)
    """
    if not os.path.exists(path):
        return [], []
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    pets  = [_deserialize_pet(p)  for p in payload.get("pets",  [])]
    tasks = [_deserialize_task(t) for t in payload.get("tasks", [])]
    return pets, tasks
