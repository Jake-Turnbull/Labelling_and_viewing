"""Flat CSV exports for analysis outside FSLeyes."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .dataset import Dataset
from .review import LABEL_FIELDS


def _atomic_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def export_project_csvs(
    dataset: Dataset, state: dict[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    """Export one row per volume and one row per subject, including pending data."""

    output_dir = Path(output_dir)
    volume_path = output_dir / "diffusion_qc_volumes.csv"
    subject_path = output_dir / "diffusion_qc_subjects.csv"
    ranking = {entry.volume: entry for entry in dataset.ranking}
    volume_columns = [
        "SubjectID",
        "Volume",
        "Rank",
        "Label",
        "Status",
        "Year",
        "Month",
        "Group",
        "Ylin",
        "BValue",
        "n_bad",
        "average_minus_ylin",
    ]
    subject_columns = [
        "SubjectID",
        "Year",
        "Month",
        "Group",
        "Ylin",
        "GoodVolumes",
        "BadVolumes",
        "UncertainVolumes",
        "UninspectedVolumes",
        "FinalQCStatus",
        "PassedScreeningRule",
    ]
    volume_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []

    for subject in dataset.subjects:
        subject_state = state["subjects"][subject.subject_id]
        good = set(subject_state["good_volumes"])
        bad = set(subject_state["bad_volumes"])
        uncertain = set(subject_state["uncertain_volumes"])
        explicitly_labelled = good | bad | uncertain
        threshold = int(state.get("settings", {}).get("good_threshold", 5))
        passed_screening = not bad and all(
            entry.volume in good for entry in dataset.ranking[:threshold]
        )
        bvalues: tuple[float, ...] = ()
        if subject.bval_path.is_file():
            try:
                bvalues = tuple(
                    float(value)
                    for value in subject.bval_path.read_text(encoding="utf-8").split()
                )
            except (OSError, ValueError):
                bvalues = ()

        for volume in range(dataset.n_volumes):
            if volume in good:
                label, status = "Good", "Reviewed"
            elif volume in bad:
                label, status = "Bad", "Reviewed"
            elif volume in uncertain:
                label, status = "Uncertain", "Unresolved"
            else:
                label = "Not inspected"
                status = (
                    "Inferred low risk"
                    if subject_state["status"] == "screened_good"
                    else "Pending"
                )
            rank_entry = ranking.get(volume)
            volume_rows.append(
                {
                    "SubjectID": subject.subject_id,
                    "Volume": volume,
                    "Rank": rank_entry.rank if rank_entry else "",
                    "Label": label,
                    "Status": status,
                    "Year": subject.year,
                    "Month": subject.month,
                    "Group": subject.group,
                    "Ylin": subject.ylin,
                    "BValue": bvalues[volume] if volume < len(bvalues) else "",
                    "n_bad": rank_entry.n_bad if rank_entry else "",
                    "average_minus_ylin": (
                        rank_entry.average_minus_ylin if rank_entry else ""
                    ),
                }
            )

        subject_rows.append(
            {
                "SubjectID": subject.subject_id,
                "Year": subject.year,
                "Month": subject.month,
                "Group": subject.group,
                "Ylin": subject.ylin,
                "GoodVolumes": len(good),
                "BadVolumes": len(bad),
                "UncertainVolumes": len(uncertain),
                "UninspectedVolumes": dataset.n_volumes - len(explicitly_labelled),
                "FinalQCStatus": subject_state["status"],
                "PassedScreeningRule": "Yes" if passed_screening else "No",
            }
        )

    _atomic_csv(volume_path, volume_columns, volume_rows)
    _atomic_csv(subject_path, subject_columns, subject_rows)
    return volume_path, subject_path