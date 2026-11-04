import csv
from pathlib import Path

from fsleyes_diffusion_qc.dataset import load_dataset
from fsleyes_diffusion_qc.export import export_project_csvs
from fsleyes_diffusion_qc.review import ReviewManager
from fsleyes_diffusion_qc.synthetic import create_synthetic_dataset


def test_synthetic_generator_creates_loadable_fixture(tmp_path: Path):
    fixture_dir = create_synthetic_dataset(tmp_path / "fixture", n_subjects=2, seed=13)

    dataset = load_dataset(
        fixture_dir / "synthetic_subjects.csv",
        fixture_dir / "synthetic_volume_ranking.csv",
        fixture_dir,
    )

    assert [subject.subject_id for subject in dataset.subjects] == ["sub-01", "sub-02"]
    assert dataset.n_volumes == 105
    assert all(subject.loading_error is None for subject in dataset.subjects)
    assert all(subject.bvec_path is not None for subject in dataset.subjects)


def test_volume_and_subject_exports_distinguish_inferred_volumes(tmp_path: Path):
    fixture_dir = create_synthetic_dataset(tmp_path / "source", n_subjects=2)
    dataset = load_dataset(
        fixture_dir / "synthetic_subjects.csv",
        fixture_dir / "synthetic_volume_ranking.csv",
        fixture_dir,
    )
    review = ReviewManager(dataset, good_threshold=5)
    subject_id = dataset.subjects[0].subject_id
    for entry in dataset.ranking[:5]:
        review.record_decision(subject_id, entry.volume, "Good")

    volume_path, subject_path = export_project_csvs(
        dataset, review.state, tmp_path / "exports"
    )
    with volume_path.open(newline="", encoding="utf-8") as stream:
        volume_rows = list(csv.DictReader(stream))
    with subject_path.open(newline="", encoding="utf-8") as stream:
        subject_rows = list(csv.DictReader(stream))

    assert len(volume_rows) == 2 * 105
    reviewed = next(
        row
        for row in volume_rows
        if row["SubjectID"] == subject_id
        and int(row["Volume"]) == dataset.ranking[0].volume
    )
    inferred = next(
        row
        for row in volume_rows
        if row["SubjectID"] == subject_id
        and int(row["Volume"]) not in {entry.volume for entry in dataset.ranking[:5]}
    )
    summary = next(row for row in subject_rows if row["SubjectID"] == subject_id)

    assert reviewed["Label"] == "Good"
    assert reviewed["Status"] == "Reviewed"
    assert inferred["Label"] == "Not inspected"
    assert inferred["Status"] == "Inferred low risk"
    assert summary["FinalQCStatus"] == "screened_good"
    assert summary["UninspectedVolumes"] == "100"