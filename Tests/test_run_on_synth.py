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


def test_subject_id_filtering(tmp_path: Path, fake_nifti_loader):
    """Only subjects matching subject_ids should be loaded."""
    root_dir = tmp_path / "dataset"
    _make_subject(root_dir, "sub-01")
    _make_subject(root_dir, "sub-02")
    _make_subject(root_dir, "sub-03")

    viewer = Image_label_and_view.diffusion_image_viewer(
        root_dir, subject_ids=["sub-01", "sub-03"], show=False
    )
    try:
        names = [s.name for s in viewer.subjects]
        assert names == ["sub-01", "sub-03"]
        assert len(viewer.subjects) == 2
    finally:
        plt.close(viewer.fig)


def test_subject_id_filtering_unknown_ids_raises(tmp_path: Path):
    """Requesting subject IDs that don't exist should raise FileNotFoundError."""
    root_dir = tmp_path / "dataset"
    _make_subject(root_dir, "sub-01")
    with pytest.raises(FileNotFoundError):
        Image_label_and_view.collect_subjects([root_dir], subject_ids=["sub-99"])


def test_max_subjects_fallback(tmp_path: Path, fake_nifti_loader):
    """max_subjects caps scanning when no subject_ids are provided."""
    root_dir = tmp_path / "dataset"
    for i in range(1, 6):
        _make_subject(root_dir, f"sub-0{i}")

    viewer = Image_label_and_view.diffusion_image_viewer(
        root_dir, max_subjects=3, show=False
    )
    try:
        assert len(viewer.subjects) == 3
        # Subjects are scanned in alphabetical order; first 3 should be loaded.
        assert [s.name for s in viewer.subjects] == ["sub-01", "sub-02", "sub-03"]
    finally:
        plt.close(viewer.fig)


def test_max_subjects_ignored_when_subject_ids_given(tmp_path: Path, fake_nifti_loader):
    """max_subjects is ignored when subject_ids is specified explicitly."""
    root_dir = tmp_path / "dataset"
    for i in range(1, 6):
        _make_subject(root_dir, f"sub-0{i}")

    # Explicitly request 4 subjects; max_subjects=2 must be ignored.
    subjects = Image_label_and_view.collect_subjects(
        [root_dir],
        subject_ids=["sub-01", "sub-02", "sub-03", "sub-04"],
        max_subjects=2,
    )
    assert len(subjects) == 4


def test_five_consecutive_good_votes_advance_subject(tmp_path: Path, fake_nifti_loader):
    """After _GOOD_STREAK_THRESHOLD consecutive good votes the viewer advances to
    the next subject and resets the streak counter."""
    root_dir = tmp_path / "dataset"
    _make_dataset(root_dir)  # sub-01 and sub-02

    viewer = Image_label_and_view.diffusion_image_viewer(root_dir, show=False)
    try:
        assert viewer.current_subject_idx == 0

        threshold = Image_label_and_view._GOOD_STREAK_THRESHOLD
        # Vote good (threshold - 1) times: should stay on the same subject.
        for _ in range(threshold - 1):
            viewer._review_and_advance("good")
        assert viewer.current_subject_idx == 0
        assert viewer._consecutive_good_count == threshold - 1

        # One more good vote reaches the threshold: subject advances.
        viewer._review_and_advance("good")
        assert viewer.current_subject_idx == 1
        assert viewer._consecutive_good_count == 0
    finally:
        plt.close(viewer.fig)


def test_bad_vote_resets_good_streak(tmp_path: Path, fake_nifti_loader):
    """A bad vote resets the consecutive-good counter."""
    root_dir = tmp_path / "dataset"
    _make_dataset(root_dir)

    viewer = Image_label_and_view.diffusion_image_viewer(root_dir, show=False)
    try:
        for _ in range(3):
            viewer._review_and_advance("good")
        assert viewer._consecutive_good_count == 3

        viewer._review_and_advance("bad")
        assert viewer._consecutive_good_count == 0

        # Streak starts fresh after the bad vote.
        viewer._review_and_advance("good")
        assert viewer._consecutive_good_count == 1
        assert viewer.current_subject_idx == 0
    finally:
        plt.close(viewer.fig)


def test_good_streak_resets_on_manual_subject_change(tmp_path: Path, fake_nifti_loader):
    """Manually switching subjects via the keyboard resets the good streak."""
    root_dir = tmp_path / "dataset"
    _make_dataset(root_dir)

    viewer = Image_label_and_view.diffusion_image_viewer(root_dir, show=False)
    try:
        for _ in range(3):
            viewer._review_and_advance("good")
        assert viewer._consecutive_good_count == 3

        viewer._load_subject(1)
        assert viewer._consecutive_good_count == 0
    finally:
        plt.close(viewer.fig)


def test_fixed_wl_check_exists_and_defaults_to_off(tmp_path: Path, fake_nifti_loader):
    """The Fix WL checkbox is present and defaults to the off (auto) state."""
    root_dir = tmp_path / "dataset"
    _make_dataset(root_dir)

    viewer = Image_label_and_view.diffusion_image_viewer(root_dir, show=False)
    try:
        assert viewer.fixed_wl_check is not None
        assert viewer._fixed_wl_on() is False
    finally:
        plt.close(viewer.fig)


def test_raw_min_max_boxes_initialized_to_data_range(tmp_path: Path, fake_nifti_loader):
    """After loading, _raw_min_val/_raw_max_val and the text boxes reflect the full data range."""
    root_dir = tmp_path / "dataset"
    _make_dataset(root_dir)

    viewer = Image_label_and_view.diffusion_image_viewer(root_dir, show=False)
    try:
        # The fake data is np.arange(6*7*8*4), so min=0, max=6*7*8*4-1=1343.
        assert viewer.raw_min_box is not None
        assert viewer.raw_max_box is not None
        assert viewer._raw_min_val == pytest.approx(0.0)
        assert viewer._raw_max_val == pytest.approx(6 * 7 * 8 * 4 - 1)
        assert float(viewer.raw_min_box.text) == pytest.approx(0.0)
        assert float(viewer.raw_max_box.text) == pytest.approx(6 * 7 * 8 * 4 - 1)
    finally:
        plt.close(viewer.fig)


def test_fixed_wl_applies_textbox_clim(tmp_path: Path, fake_nifti_loader):
    """When Fix WL is active, all orthogonal views use the values from the text boxes."""
    root_dir = tmp_path / "dataset"
    _make_dataset(root_dir)

    viewer = Image_label_and_view.diffusion_image_viewer(root_dir, show=False)
    try:
        # Enable Fix WL by toggling the checkbox.
        viewer.fixed_wl_check.set_active(0)
        assert viewer._fixed_wl_on() is True

        # Set specific values directly (simulates the user typing and pressing Enter).
        viewer._raw_min_val = 100.0
        viewer._raw_max_val = 500.0
        viewer.update_display()

        for im in (viewer.im_axial, viewer.im_sagittal, viewer.im_coronal):
            clim = im.get_clim()
            assert clim[0] == pytest.approx(100.0)
            assert clim[1] == pytest.approx(500.0)
    finally:
        plt.close(viewer.fig)
