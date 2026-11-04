"""Input CSV and diffusion-file validation, independent of FSLeyes."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class InputValidationError(ValueError):
    """Raised when an input CSV cannot define a valid review dataset."""


@dataclass(frozen=True)
class RankingEntry:
    volume: int
    rank: int
    n_bad: int
    average_minus_ylin: float


@dataclass(frozen=True)
class Subject:
    subject_id: str
    scan_date_raw: float
    ylin: float
    year: int
    month: int
    group: str
    subject_dir: Path
    image_path: Path
    bval_path: Path
    bvec_path: Path | None
    loading_error: str | None = None


@dataclass(frozen=True)
class Dataset:
    subjects: tuple[Subject, ...]
    ranking: tuple[RankingEntry, ...]
    n_volumes: int


_SUBJECT_COLUMNS = {"SubjectID", "ScanDate", "Ylin", "Year", "Month", "Group"}
_RANKING_COLUMNS = {"volume", "n_bad", "average_minus_ylin"}


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing = required - fields
            if missing:
                raise InputValidationError(
                    f"{path} is missing required columns: {', '.join(sorted(missing))}"
                )
            return list(reader)
    except OSError as exc:
        raise InputValidationError(f"Cannot read CSV {path}: {exc}") from exc


def _number(row: dict[str, str], column: str, source: Path, row_number: int) -> float:
    try:
        value = float(row[column])
    except (TypeError, ValueError, KeyError) as exc:
        raise InputValidationError(
            f"{source}, row {row_number}: {column} must be numeric"
        ) from exc
    if not math.isfinite(value):
        raise InputValidationError(f"{source}, row {row_number}: {column} must be finite")
    return value


def _image_paths(subject_dir: Path) -> tuple[Path, Path, Path | None]:
    raw_dir = subject_dir / "dMRI" / "raw"
    compressed = raw_dir / "AP.nii.gz"
    uncompressed = raw_dir / "AP.nii"
    image_path = compressed if compressed.is_file() else uncompressed
    return image_path, raw_dir / "AP.bval", raw_dir / "AP.bvec"


def _validate_imaging_files(
    image_path: Path, bval_path: Path, expected_volumes: int
) -> str | None:
    if not image_path.is_file():
        return f"Missing diffusion image: {image_path}"
    if not bval_path.is_file():
        return f"Missing b-values: {bval_path}"

    try:
        import nibabel as nib

        image = nib.load(str(image_path))
        if len(image.shape) != 4:
            return f"Expected a 4-D NIfTI, found shape {image.shape}"
        if image.shape[-1] != expected_volumes:
            return (
                f"Expected {expected_volumes} NIfTI volumes, found {image.shape[-1]}"
            )
    except Exception as exc:
        return f"Cannot read NIfTI {image_path}: {exc}"

    try:
        values = [float(value) for value in bval_path.read_text(encoding="utf-8").split()]
    except (OSError, ValueError) as exc:
        return f"Cannot read b-values {bval_path}: {exc}"
    if len(values) != expected_volumes:
        return f"Expected {expected_volumes} b-values, found {len(values)}"
    if not all(math.isfinite(value) for value in values):
        return f"B-values contain a non-finite value: {bval_path}"
    if any(value < 0 for value in values):
        return f"B-values cannot be negative: {bval_path}"
    return None


def load_dataset(
    subject_csv: str | Path,
    ranking_csv: str | Path,
    subject_root: str | Path,
) -> Dataset:
    """Load CSV rows in source order and validate matching imaging files.

    Image failures are retained on each subject as ``loading_error`` so one
    unreadable scan does not prevent the rest of the project from opening.
    """

    subject_csv = Path(subject_csv)
    ranking_csv = Path(ranking_csv)
    subject_root = Path(subject_root)
    subject_rows = _read_rows(subject_csv, _SUBJECT_COLUMNS)
    ranking_rows = _read_rows(ranking_csv, _RANKING_COLUMNS)

    subjects: list[Subject] = []
    seen_subject_ids: set[str] = set()
    for row_number, row in enumerate(subject_rows, start=2):
        subject_id = (row.get("SubjectID") or "").strip()
        if not subject_id:
            raise InputValidationError(f"{subject_csv}, row {row_number}: SubjectID is empty")
        if subject_id in seen_subject_ids:
            raise InputValidationError(
                f"{subject_csv}, row {row_number}: duplicate SubjectID {subject_id!r}"
            )
        seen_subject_ids.add(subject_id)

        year_value = _number(row, "Year", subject_csv, row_number)
        month_value = _number(row, "Month", subject_csv, row_number)
        if not year_value.is_integer() or not month_value.is_integer():
            raise InputValidationError(
                f"{subject_csv}, row {row_number}: Year and Month must be integers"
            )
        year, month = int(year_value), int(month_value)
        if not 1 <= month <= 12:
            raise InputValidationError(f"{subject_csv}, row {row_number}: Month must be 1-12")

        group = (row.get("Group") or "").strip().upper()
        if group not in {"HIGH", "LOW"}:
            raise InputValidationError(
                f"{subject_csv}, row {row_number}: Group must be HIGH or LOW"
            )

        subject_dir = subject_root / subject_id
        image_path, bval_path, bvec_path = _image_paths(subject_dir)
        subjects.append(
            Subject(
                subject_id=subject_id,
                scan_date_raw=_number(row, "ScanDate", subject_csv, row_number),
                ylin=_number(row, "Ylin", subject_csv, row_number),
                year=year,
                month=month,
                group=group,
                subject_dir=subject_dir,
                image_path=image_path,
                bval_path=bval_path,
                bvec_path=bvec_path if bvec_path.is_file() else None,
            )
        )

    if not subjects:
        raise InputValidationError(f"{subject_csv} contains no subjects")

    ranking: list[RankingEntry] = []
    seen_volumes: set[int] = set()
    for rank, row in enumerate(ranking_rows, start=1):
        row_number = rank + 1
        raw_volume = _number(row, "volume", ranking_csv, row_number)
        raw_n_bad = _number(row, "n_bad", ranking_csv, row_number)
        if not raw_volume.is_integer() or raw_volume < 0:
            raise InputValidationError(
                f"{ranking_csv}, row {row_number}: volume must be a non-negative integer"
            )
        if not raw_n_bad.is_integer() or raw_n_bad < 0:
            raise InputValidationError(
                f"{ranking_csv}, row {row_number}: n_bad must be a non-negative integer"
            )
        volume = int(raw_volume)
        if volume in seen_volumes:
            raise InputValidationError(
                f"{ranking_csv}, row {row_number}: duplicate volume index {volume}"
            )
        seen_volumes.add(volume)
        ranking.append(
            RankingEntry(
                volume=volume,
                rank=rank,
                n_bad=int(raw_n_bad),
                average_minus_ylin=_number(
                    row, "average_minus_ylin", ranking_csv, row_number
                ),
            )
        )

    if not ranking:
        raise InputValidationError(f"{ranking_csv} contains no ranked volumes")
    n_volumes = max(entry.volume for entry in ranking) + 1
    if len(ranking) != n_volumes:
        raise InputValidationError("Ranking CSV must include every volume index from 0 to N-1")

    validated_subjects: list[Subject] = []
    for subject in subjects:
        error = _validate_imaging_files(subject.image_path, subject.bval_path, n_volumes)
        validated_subjects.append(
            Subject(**{**subject.__dict__, "loading_error": error})
        )

    return Dataset(tuple(validated_subjects), tuple(ranking), n_volumes)


def read_b_values(path: str | Path) -> tuple[float, ...]:
    """Read b-values for display after dataset validation."""

    values = tuple(float(value) for value in Path(path).read_text().split())
    return values