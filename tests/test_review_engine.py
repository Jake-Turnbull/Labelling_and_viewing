import csv
import copy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fsleyes_diffusion_qc.dataset import load_dataset
from fsleyes_diffusion_qc.review import ReviewManager
from fsleyes_diffusion_qc.storage import ProjectLock, ProjectLockedError, ProjectStore


RANKED_FIRST = [46, 60, 92, 79, 69]


@pytest.fixture
def project_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    subject_root = tmp_path / "subjects"
    subject_root.mkdir()
    subject_csv = tmp_path / "subjects.csv"
    ranking_csv = tmp_path / "ranking.csv"

    with subject_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["SubjectID", "ScanDate", "Ylin", "Year", "Month", "Group"])
        for index in range(1, 6):
            subject_id = f"00{index}"
            writer.writerow([subject_id, "2022.057534", "0.2", 2022, index, "HIGH"])
            raw_dir = subject_root / subject_id / "dMRI" / "raw"
            raw_dir.mkdir(parents=True)
            image = nib.Nifti1Image(np.zeros((2, 2, 2, 105), dtype=np.uint8), np.eye(4))
            nib.save(image, raw_dir / "AP.nii.gz")
            (raw_dir / "AP.bval").write_text(" ".join(["1000"] * 105), encoding="utf-8")

    ranked_volumes = RANKED_FIRST + [
        volume for volume in range(105) if volume not in RANKED_FIRST
    ]
    with ranking_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["volume", "n_bad", "average_minus_ylin"])
        for rank, volume in enumerate(ranked_volumes):
            writer.writerow([volume, 105 - rank, -0.5 + rank / 1000])

    return subject_csv, ranking_csv, subject_root


def test_load_dataset_preserves_csv_order_and_string_subject_ids(project_inputs):
    subject_csv, ranking_csv, subject_root = project_inputs

    dataset = load_dataset(subject_csv, ranking_csv, subject_root)

    assert [subject.subject_id for subject in dataset.subjects] == [
        "001",
        "002",
        "003",
        "004",
        "005",
    ]
    assert [entry.volume for entry in dataset.ranking[:5]] == RANKED_FIRST
    assert dataset.n_volumes == 105
    assert all(subject.loading_error is None for subject in dataset.subjects)


def test_only_first_ranked_threshold_good_volumes_screen_subject(project_inputs):
    subject_csv, ranking_csv, subject_root = project_inputs
    dataset = load_dataset(subject_csv, ranking_csv, subject_root)
    review = ReviewManager(dataset, good_threshold=5)
    subject_id = "001"

    for volume in RANKED_FIRST:
        status = review.record_decision(subject_id, volume, "Good")

    assert status == "screened_good"
    assert review.next_volume(subject_id) is None
    assert review.uninspected_volumes(subject_id)


def test_bad_prevents_screening_and_requires_all_volumes_resolved(project_inputs):
    subject_csv, ranking_csv, subject_root = project_inputs
    dataset = load_dataset(subject_csv, ranking_csv, subject_root)
    review = ReviewManager(dataset, good_threshold=5)
    subject_id = "001"

    review.record_decision(subject_id, RANKED_FIRST[0], "Bad")
    for volume in RANKED_FIRST[1:]:
        review.record_decision(subject_id, volume, "Good")

    assert review.subject_status(subject_id) == "in_progress"
    for volume in range(dataset.n_volumes):
        if volume not in RANKED_FIRST:
            review.record_decision(subject_id, volume, "Good")
    assert review.subject_status(subject_id) == "fully_reviewed"
    assert review.next_volume(subject_id) is None


def test_uncertain_and_skipped_volumes_are_revisited_after_unresolved_volumes(project_inputs):
    subject_csv, ranking_csv, subject_root = project_inputs
    dataset = load_dataset(subject_csv, ranking_csv, subject_root)
    review = ReviewManager(dataset, good_threshold=5)
    subject_id = "001"
    first_volume = RANKED_FIRST[0]
    second_volume = RANKED_FIRST[1]

    review.record_decision(subject_id, first_volume, "Uncertain")
    assert review.next_volume(subject_id) == second_volume
    review.skip_volume(subject_id, second_volume)
    assert review.next_volume(subject_id) == RANKED_FIRST[2]

    deferred_volumes = {first_volume, second_volume}
    for volume in range(dataset.n_volumes):
        if volume not in deferred_volumes:
            review.record_decision(subject_id, volume, "Good")

    assert review.next_volume(subject_id) == first_volume
    assert second_volume in review.state["subjects"][subject_id]["skipped_volumes"]


def test_deferred_label_outside_screen_blocks_early_completion(project_inputs):
    subject_csv, ranking_csv, subject_root = project_inputs
    dataset = load_dataset(subject_csv, ranking_csv, subject_root)
    review = ReviewManager(dataset, good_threshold=5)
    subject_id = "001"
    deferred_volume = dataset.ranking[-1].volume

    review.record_decision(subject_id, deferred_volume, "Uncertain")
    for entry in dataset.ranking[:5]:
        review.record_decision(subject_id, entry.volume, "Good")
    for entry in dataset.ranking[5:-1]:
        review.record_decision(subject_id, entry.volume, "Good")

    assert review.subject_status(subject_id) == "in_progress"
    assert review.next_volume(subject_id) == deferred_volume


def test_changing_screened_good_to_bad_reopens_subject(project_inputs):
    subject_csv, ranking_csv, subject_root = project_inputs
    dataset = load_dataset(subject_csv, ranking_csv, subject_root)
    review = ReviewManager(dataset, good_threshold=5)
    subject_id = "001"
    for volume in RANKED_FIRST:
        review.record_decision(subject_id, volume, "Good")

    review.record_decision(subject_id, RANKED_FIRST[0], "Bad")

    assert review.subject_status(subject_id) == "in_progress"
    assert review.next_volume(subject_id) == dataset.ranking[5].volume


def test_resume_restores_exact_subject_and_rank(project_inputs):
    subject_csv, ranking_csv, subject_root = project_inputs
    dataset = load_dataset(subject_csv, ranking_csv, subject_root)
    review = ReviewManager(dataset, good_threshold=5)
    review.select_volume("001", dataset.ranking[6].volume)

    resumed = ReviewManager(dataset, copy.deepcopy(review.state), good_threshold=5)

    assert resumed.current_subject_id == "001"
    assert resumed.state["progress"]["current_rank"] == 7


def test_selecting_completed_subject_starts_at_first_rank(project_inputs):
    subject_csv, ranking_csv, subject_root = project_inputs
    dataset = load_dataset(subject_csv, ranking_csv, subject_root)
    review = ReviewManager(dataset, good_threshold=5)
    for entry in dataset.ranking[:5]:
        review.record_decision("001", entry.volume, "Good")
    review.select_volume("002", dataset.ranking[12].volume)

    review.select_subject("001")

    assert review.state["progress"]["current_rank"] == 1


def test_missing_imaging_is_a_loading_error(project_inputs):
    subject_csv, ranking_csv, subject_root = project_inputs
    (subject_root / "001" / "dMRI" / "raw" / "AP.nii.gz").unlink()

    dataset = load_dataset(subject_csv, ranking_csv, subject_root)
    review = ReviewManager(dataset)

    assert review.subject_status("001") == "loading_error"
    assert review.next_volume("001") is None


def test_store_recovers_backup_and_lock_excludes_second_writer(tmp_path: Path):
    project_path = tmp_path / "project.json"
    store = ProjectStore(project_path)
    original = {"schema_version": 1, "subjects": {"001": {"notes": "saved"}}}
    store.save(original)
    store.save({"schema_version": 1, "subjects": {"001": {"notes": "new"}}})
    project_path.write_text("{broken", encoding="utf-8")

    assert store.load() == original
    assert ProjectStore(project_path).load() == original

    first_lock = ProjectLock(project_path)
    second_lock = ProjectLock(project_path)
    first_lock.acquire()
    try:
        with pytest.raises(ProjectLockedError):
            second_lock.acquire()
    finally:
        first_lock.release()
    second_lock.acquire()
    second_lock.release()