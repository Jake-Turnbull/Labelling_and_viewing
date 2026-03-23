import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ViewerScripts import Image_label_and_view


def _make_subject(root_dir: Path, subject_id: str, *, volumes: int = 4) -> None:
    subject_dir = root_dir / subject_id / "dMRI" / "raw"
    subject_dir.mkdir(parents=True, exist_ok=True)
    (subject_dir / "AP.nii.gz").touch()


def _make_dataset(root_dir: Path) -> None:
    _make_subject(root_dir, "sub-01")
    _make_subject(root_dir, "sub-02")


def _read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


@pytest.fixture
def fake_nifti_loader(monkeypatch):
    def _loader(_path: str) -> np.ndarray:
        return np.arange(6 * 7 * 8 * 4, dtype=np.float32).reshape(6, 7, 8, 4).copy()

    monkeypatch.setattr(Image_label_and_view, "load_subject_nifti", _loader)


def test_run_on_synth(tmp_path: Path, fake_nifti_loader):
    root_dir = tmp_path / "synthetic_dwi_dataset"
    _make_dataset(root_dir)

    viewer = Image_label_and_view.diffusion_image_viewer(root_dir, show=False)
    try:
        assert viewer.vol == 1
        assert viewer.current_subject_idx == 0
        assert _read_log(root_dir / "diffusion_review_log.jsonl") == []

        viewer._review_and_advance("good")
        assert viewer.vol == 2
        assert viewer.current_subject_idx == 0
        assert _read_log(root_dir / "diffusion_review_log.jsonl") == []

        viewer._review_and_advance("bad")
        assert viewer.vol == 3
        assert viewer.current_subject_idx == 0
        assert _read_log(root_dir / "diffusion_review_log.jsonl") == [
            {"subject_id": "sub-01", "volume": [2]}
        ]

        viewer._review_and_advance("bad")
        assert viewer.vol == 4
        assert _read_log(root_dir / "diffusion_review_log.jsonl") == [
            {"subject_id": "sub-01", "volume": [2, 3]}
        ]
    finally:
        plt.close(viewer.fig)


def test_existing_log_is_normalized_and_slice_sliders_are_stacked(tmp_path: Path, fake_nifti_loader):
    root_dir = tmp_path / "synthetic_dwi_dataset"
    _make_dataset(root_dir)
    log_path = root_dir / "diffusion_review_log.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2026-03-23T16:23:27+00:00", "event": "view", "subject_id": "sub-01"}),
                json.dumps({"timestamp": "2026-03-23T16:23:28+00:00", "event": "bad", "subject_id": "sub-01", "volume": 1}),
                json.dumps({"timestamp": "2026-03-23T16:23:29+00:00", "event": "bad", "subject_id": "sub-01", "volume": 3}),
            ]
        )
        + "\n"
    )

    viewer = Image_label_and_view.diffusion_image_viewer(root_dir, show=False)
    try:
        assert _read_log(log_path) == [{"subject_id": "sub-01", "volume": [1, 3]}]

        axial_pos = viewer.axial_slider.ax.get_position()
        sagittal_pos = viewer.sagittal_slider.ax.get_position()
        coronal_pos = viewer.coronal_slider.ax.get_position()
        vol_pos = viewer.vol_slider.ax.get_position()

        assert axial_pos.x0 == sagittal_pos.x0 == coronal_pos.x0
        assert axial_pos.width == sagittal_pos.width == coronal_pos.width
        assert axial_pos.width < vol_pos.width
    finally:
        plt.close(viewer.fig)
