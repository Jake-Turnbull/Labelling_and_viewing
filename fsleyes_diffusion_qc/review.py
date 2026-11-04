"""Review state machine for ranked, volume-level diffusion QC."""

from __future__ import annotations

from typing import Any

from .dataset import Dataset


LABEL_FIELDS = {
    "Good": "good_volumes",
    "Bad": "bad_volumes",
    "Uncertain": "uncertain_volumes",
}


def _empty_subject(subject: Any) -> dict[str, Any]:
    return {
        "acquisition": {
            "year": subject.year,
            "month": subject.month,
            "scan_date_raw": subject.scan_date_raw,
        },
        "original_group": subject.group,
        "original_ylin": subject.ylin,
        "status": "loading_error" if subject.loading_error else "pending",
        "loading_error": subject.loading_error,
        "good_volumes": [],
        "bad_volumes": [],
        "uncertain_volumes": [],
        "skipped_volumes": [],
        "notes": "",
    }


class ReviewManager:
    """Mutate QC decisions and derive progression without a GUI dependency."""

    def __init__(
        self,
        dataset: Dataset,
        state: dict[str, Any] | None = None,
        *,
        good_threshold: int = 5,
    ) -> None:
        if good_threshold < 1:
            raise ValueError("good_threshold must be at least 1")
        self.dataset = dataset
        has_saved_state = state is not None
        self._subjects = {subject.subject_id: subject for subject in dataset.subjects}
        self._ranked_volumes = [entry.volume for entry in dataset.ranking]
        self._rank_by_volume = {entry.volume: entry.rank for entry in dataset.ranking}
        self.state = state if state is not None else self._new_state(good_threshold)
        if self.state.get("schema_version", 1) != 1:
            raise ValueError(
                f"Unsupported QC project schema version: {self.state.get('schema_version')}"
            )
        self.state.setdefault("schema_version", 1)
        self.state.setdefault("settings", {})
        self.state["settings"].setdefault("good_threshold", good_threshold)
        self.state["settings"].setdefault("volume_index_base", 0)
        self.state.setdefault("progress", {"current_subject": None, "current_rank": 1})
        self.state.setdefault("subjects", {})
        for subject in dataset.subjects:
            subject_state = self.state["subjects"].setdefault(
                subject.subject_id, _empty_subject(subject)
            )
            subject_state.setdefault("loading_error", subject.loading_error)
            for field in ("good_volumes", "bad_volumes", "uncertain_volumes", "skipped_volumes"):
                subject_state.setdefault(field, [])
            self._refresh_status(subject.subject_id)

        if self.state["progress"].get("current_subject") not in self._subjects:
            self.state["progress"]["current_subject"] = self._first_incomplete_subject()
            self._set_progress_rank()
        elif has_saved_state:
            try:
                saved_rank = int(self.state["progress"].get("current_rank", 1))
            except (TypeError, ValueError):
                saved_rank = 1
            self.state["progress"]["current_rank"] = max(
                1, min(saved_rank, len(self.dataset.ranking))
            )
        else:
            self._set_progress_rank()

    def _new_state(self, good_threshold: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "settings": {"good_threshold": good_threshold, "volume_index_base": 0},
            "progress": {"current_subject": None, "current_rank": 1},
            "subjects": {},
        }

    def _subject_state(self, subject_id: str) -> dict[str, Any]:
        try:
            return self.state["subjects"][subject_id]
        except KeyError as exc:
            raise KeyError(f"Unknown subject ID: {subject_id}") from exc

    def _volumes(self, subject_id: str, field: str) -> set[int]:
        return set(self._subject_state(subject_id)[field])

    def _threshold_passed(self, subject_id: str) -> bool:
        threshold = int(self.state["settings"]["good_threshold"])
        if (
            self._volumes(subject_id, "bad_volumes")
            or self._volumes(subject_id, "uncertain_volumes")
            or self._volumes(subject_id, "skipped_volumes")
        ):
            return False
        return all(
            volume in self._volumes(subject_id, "good_volumes")
            for volume in self._ranked_volumes[:threshold]
        )

    def _all_volumes_resolved(self, subject_id: str) -> bool:
        decided = self._volumes(subject_id, "good_volumes") | self._volumes(
            subject_id, "bad_volumes"
        )
        return len(decided) == self.dataset.n_volumes

    def _refresh_status(self, subject_id: str) -> str:
        subject_state = self._subject_state(subject_id)
        if subject_state.get("loading_error"):
            status = "loading_error"
        elif self._all_volumes_resolved(subject_id):
            status = "fully_reviewed"
        elif self._threshold_passed(subject_id):
            status = "screened_good"
        elif any(
            self._volumes(subject_id, field)
            for field in ("good_volumes", "bad_volumes", "uncertain_volumes", "skipped_volumes")
        ):
            status = "in_progress"
        else:
            status = "pending"
        subject_state["status"] = status
        return status

    def _first_incomplete_subject(self) -> str | None:
        for subject in self.dataset.subjects:
            status = self._refresh_status(subject.subject_id)
            if status not in {"screened_good", "fully_reviewed", "loading_error"}:
                return subject.subject_id
        return None

    def _set_progress_rank(self) -> None:
        subject_id = self.current_subject_id
        if subject_id is None:
            self.state["progress"]["current_rank"] = 1
            return
        volume = self.next_volume(subject_id)
        if volume is not None:
            self.state["progress"]["current_rank"] = self._rank_by_volume.get(volume, 1)
        else:
            self.state["progress"]["current_rank"] = 1

    @property
    def current_subject_id(self) -> str | None:
        return self.state["progress"].get("current_subject")

    def subject_status(self, subject_id: str) -> str:
        return self._refresh_status(subject_id)

    def next_volume(self, subject_id: str) -> int | None:
        subject_state = self._subject_state(subject_id)
        if subject_state.get("loading_error") or self._refresh_status(subject_id) in {
            "screened_good",
            "fully_reviewed",
        }:
            return None
        good = self._volumes(subject_id, "good_volumes")
        bad = self._volumes(subject_id, "bad_volumes")
        unresolved = [
            volume for volume in self._ranked_volumes if volume not in good | bad
        ]
        deferred = self._volumes(subject_id, "uncertain_volumes") | self._volumes(
            subject_id, "skipped_volumes"
        )
        return next(
            (volume for volume in unresolved if volume not in deferred),
            unresolved[0] if unresolved else None,
        )

    def select_subject(self, subject_id: str) -> None:
        self._subject_state(subject_id)
        self.state["progress"]["current_subject"] = subject_id
        self._set_progress_rank()

    def select_volume(self, subject_id: str, volume: int) -> None:
        self._subject_state(subject_id)
        if not 0 <= volume < self.dataset.n_volumes:
            raise ValueError(f"Volume index {volume} is out of bounds")
        self.state["progress"]["current_subject"] = subject_id
        self.state["progress"]["current_rank"] = self._rank_by_volume.get(volume, 1)

    def mark_loading_error(self, subject_id: str, message: str) -> None:
        subject_state = self._subject_state(subject_id)
        subject_state["loading_error"] = message
        self._refresh_status(subject_id)

    def clear_loading_error(self, subject_id: str) -> None:
        subject_state = self._subject_state(subject_id)
        subject_state["loading_error"] = None
        self._refresh_status(subject_id)

    def record_decision(self, subject_id: str, volume: int, label: str) -> str:
        if label not in LABEL_FIELDS:
            raise ValueError(f"Unsupported label: {label}")
        if not 0 <= volume < self.dataset.n_volumes:
            raise ValueError(f"Volume index {volume} is out of bounds")
        subject_state = self._subject_state(subject_id)
        if subject_state.get("loading_error"):
            raise ValueError("Cannot label a subject with a loading error")

        for field in (*LABEL_FIELDS.values(), "skipped_volumes"):
            subject_state[field] = [item for item in subject_state[field] if item != volume]
        subject_state[LABEL_FIELDS[label]].append(volume)
        for field in LABEL_FIELDS.values():
            subject_state[field].sort()
        self.state["progress"]["current_subject"] = subject_id
        self._refresh_status(subject_id)
        next_volume = self.next_volume(subject_id)
        if next_volume is not None:
            self.state["progress"]["current_rank"] = self._rank_by_volume[next_volume]
        return subject_state["status"]

    def skip_volume(self, subject_id: str, volume: int) -> None:
        if not 0 <= volume < self.dataset.n_volumes:
            raise ValueError(f"Volume index {volume} is out of bounds")
        subject_state = self._subject_state(subject_id)
        for field in LABEL_FIELDS.values():
            subject_state[field] = [item for item in subject_state[field] if item != volume]
        if volume not in subject_state["skipped_volumes"]:
            subject_state["skipped_volumes"].append(volume)
            subject_state["skipped_volumes"].sort()
        self.state["progress"]["current_subject"] = subject_id
        self._refresh_status(subject_id)
        next_volume = self.next_volume(subject_id)
        if next_volume is not None:
            self.state["progress"]["current_rank"] = self._rank_by_volume[next_volume]

    def set_note(self, subject_id: str, note: str) -> None:
        self._subject_state(subject_id)["notes"] = note

    def advance_subject(self) -> str | None:
        """Select the next non-complete, loadable subject without wrapping."""

        subjects = list(self.dataset.subjects)
        current = self.current_subject_id
        start = next((i + 1 for i, subject in enumerate(subjects) if subject.subject_id == current), 0)
        for offset in range(len(subjects)):
            subject = subjects[(start + offset) % len(subjects)]
            if self._refresh_status(subject.subject_id) not in {
                "screened_good",
                "fully_reviewed",
                "loading_error",
            }:
                self.select_subject(subject.subject_id)
                return subject.subject_id
        self.state["progress"]["current_subject"] = None
        return None

    def uninspected_volumes(self, subject_id: str) -> list[int]:
        subject_state = self._subject_state(subject_id)
        inspected = set().union(
            *(set(subject_state[field]) for field in LABEL_FIELDS.values())
        )
        return sorted(set(range(self.dataset.n_volumes)) - inspected)