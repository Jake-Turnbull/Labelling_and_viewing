from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox


ArrayLike4D = np.ndarray
PathLike = Union[str, os.PathLike]
InputType = Union[ArrayLike4D, PathLike, Sequence[PathLike]]

# Number of consecutive "good" votes that trigger automatic subject advancement.
_GOOD_STREAK_THRESHOLD = 5

# Baseline bottom fraction for which the control positions below were designed.
# _relayout() scales them proportionally when the figure is resized.
_CTRL_BASELINE_H: float = 0.22

# Normalised [left, bottom, width, height] for every control widget, measured
# at the baseline figure height fraction _CTRL_BASELINE_H.
_CTRL_BASE_POS: dict[str, list[float]] = {
    # Left column – FFT window toggle
    "window":      [0.04, 0.130, 0.10, 0.065],
    # Middle-left – subject / volume sliders (narrower than before)
    "subject":     [0.04, 0.105, 0.25, 0.025],
    "vol":         [0.04, 0.08, 0.25, 0.025],
    # Middle – Good / Bad review buttons
    "good":        [0.44, 0.09, 0.08, 0.038],
    "bad":         [0.53, 0.09, 0.08, 0.038],
    # Right-of-middle – orthogonal-slice sliders (narrower)
    "axial":       [0.63+0.1, 0.068, 0.20, 0.018],
    "sagittal":    [0.63+0.1, 0.044, 0.20, 0.018],
    "coronal":     [0.63+0.1, 0.020, 0.20, 0.018],
    # Right column – below the info panel: Fix WL toggle + raw min/max text inputs
    "fixed_wl":    [0.63+0.1, 0.178+0.1, 0.09, 0.055],
    "raw_min_box": [0.63+0.1, 0.130+0.1, 0.24, 0.030],
    "raw_max_box": [0.63+0.1, 0.092+0.1, 0.24, 0.030],
}


@dataclass
class SubjectInfo:
    """Store the filesystem path and display name for one subject."""

    path: str
    name: str

def _folder_label(path: str) -> str:
    """Return a readable label for a subject folder."""

    label = os.path.basename(os.path.normpath(path))
    return label if label else path


def _has_ap_file(subj_path: str) -> bool:
    """Check whether a subject folder contains the expected AP NIfTI."""

    f1 = os.path.join(subj_path, "dMRI", "raw", "AP.nii")
    f2 = os.path.join(subj_path, "dMRI", "raw", "AP.nii.gz")
    return os.path.isfile(f1) or os.path.isfile(f2)


def collect_subjects(
    root_dirs: Sequence[PathLike],
    *,
    subject_ids: Optional[Sequence[str]] = None,
    max_subjects: Optional[int] = None,
) -> List[SubjectInfo]:
    """Collect all subject folders that contain an AP diffusion volume.

    Parameters
    ----------
    root_dirs:
        One or more root directories to scan for subject folders.
    subject_ids:
        When provided, only folders whose basename matches one of these IDs are
        included.  All other folders are silently skipped.
    max_subjects:
        When *subject_ids* is ``None`` this cap limits how many subjects are
        collected across all root directories combined.  Ignored when
        *subject_ids* is given.
    """

    id_set: Optional[set[str]] = set(subject_ids) if subject_ids is not None else None
    subjects: List[SubjectInfo] = []

    for root in root_dirs:
        root_dir = os.fspath(root)
        if not os.path.isdir(root_dir):
            continue

        # Case 1: the root itself is a subject folder
        if _has_ap_file(root_dir):
            label = _folder_label(root_dir)
            if id_set is None or label in id_set:
                subjects.append(SubjectInfo(path=root_dir, name=label))
            continue

        # Case 2: scan immediate child folders
        try:
            entries = sorted(
                (e for e in os.scandir(root_dir) if e.is_dir() and e.name not in {".", ".."}),
                key=lambda e: e.name,
            )
        except FileNotFoundError:
            continue

        for entry in entries:
            subj_path = entry.path
            label = _folder_label(subj_path)

            if id_set is not None:
                if label not in id_set:
                    continue
            elif max_subjects is not None and len(subjects) >= max_subjects:
                break

            if _has_ap_file(subj_path):
                subjects.append(SubjectInfo(path=subj_path, name=label))

    if not subjects:
        raise FileNotFoundError(
            "No subject folders with dMRI/raw/AP.nii or AP.nii.gz were found."
        )

    return subjects


def load_subject_nifti(subj_path: str) -> np.ndarray:
    """Load a subject AP NIfTI and normalize it to a 4-D float array."""

    try:
        import nibabel as nib
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "This script requires nibabel to load NIfTI files. Install it with: pip install nibabel"
        ) from e

    f1 = os.path.join(subj_path, "dMRI", "raw", "AP.nii")
    f2 = os.path.join(subj_path, "dMRI", "raw", "AP.nii.gz")

    if os.path.isfile(f1):
        data = nib.load(f1).get_fdata()
    elif os.path.isfile(f2):
        data = nib.load(f2).get_fdata()
    else:
        raise FileNotFoundError(f"No AP.nii or AP.nii.gz found for subject: {subj_path}")

    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 3:
        data = data[..., np.newaxis]

    if data.ndim != 4:
        raise ValueError(
            f"Loaded NIfTI has unsupported dimensionality ({data.ndim}). Expected 4-D."
        )

    return data


def _rot_display(img2d: np.ndarray) -> np.ndarray:
    """Rotate 2-D slices so they display in a consistent orientation."""

    return np.rot90(np.asarray(img2d))


def _hann2d(nx: int, ny: int) -> np.ndarray:
    """Build a separable 2-D Hann window for FFT preprocessing."""

    wx = np.hanning(nx)
    wy = np.hanning(ny)
    return np.outer(wx, wy)


def _radial_power_spectrum(power2d: np.ndarray, nbins: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Average 2-D power values into radial frequency bins."""

    nx, ny = power2d.shape
    if nbins is None:
        nbins = max(1, int(round(min(nx, ny) / 2)))

    fx = (np.arange(nx) - np.floor(nx / 2.0)) / float(nx)
    fy = (np.arange(ny) - np.floor(ny / 2.0)) / float(ny)
    FY, FX = np.meshgrid(fy, fx)
    FR = np.sqrt(FX**2 + FY**2)

    fmax = 0.5
    edges = np.linspace(0.0, fmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fr_vec = FR.ravel()
    p_vec = power2d.ravel()
    bin_idx = np.digitize(fr_vec, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < nbins)

    if not np.any(valid):
        return centers, np.zeros(nbins, dtype=np.float64)

    sums = np.bincount(bin_idx[valid], weights=p_vec[valid], minlength=nbins)
    counts = np.bincount(bin_idx[valid], minlength=nbins)
    with np.errstate(divide="ignore", invalid="ignore"):
        radial = sums / counts
    radial[counts == 0] = np.nan
    return centers, radial


class DiffusionImageViewer:
    def __init__(
        self,
        dwi_or_dir: InputType,
        *,
        subject_ids: Optional[Sequence[str]] = None,
        max_subjects: Optional[int] = None,
    ):
        """Initialize viewer state, build the UI, and load the first subject.

        Parameters
        ----------
        dwi_or_dir:
            A 4-D numpy array, a subject/root directory path, or a sequence of
            root directory paths.
        subject_ids:
            Optional list of subject folder names to include.  When given, any
            folder not in this list is ignored.
        max_subjects:
            When *subject_ids* is ``None``, stop scanning after this many
            subjects have been found.  Useful as a quick sanity-check pass
            over a large dataset.
        """

        self.subjects: List[SubjectInfo] = []
        self.current_subject_idx = -1
        self.dwi: np.ndarray | None = None
        self.root_dirs: List[str] = []
        self.review_log_path: str | None = None
        self._suppress_subject_callback = False
        self._updating_controls = False

        # Consecutive-good-vote streak counter for auto subject advancement.
        self._consecutive_good_count: int = 0

        self.vol = 1
        self.axial_slice = 0
        self.sagittal_slice = 0
        self.coronal_slice = 0

        self.fig = None
        self._main_gs = None
        self.ax_axial = None
        self.ax_sagittal = None
        self.ax_coronal = None
        self.ax_fft = None
        self.ax_radial = None
        self.ax_info = None

        self.im_axial = None
        self.im_sagittal = None
        self.im_coronal = None
        self.im_fft = None
        self.line_slice = None
        self.line_volume = None
        self.info_text = None

        # Widget references
        self.subject_slider = None
        self.vol_slider = None
        self.axial_slider = None
        self.sagittal_slider = None
        self.coronal_slider = None
        self.window_check = None
        self.fixed_wl_check = None
        self.good_button = None
        self.bad_button = None
        self.raw_min_box = None
        self.raw_max_box = None
        # Authoritative float values for the raw window level.
        self._raw_min_val: float = 0.0
        self._raw_max_val: float = 1.0

        # Internal dict of control axes for relayout.
        self._ctrl_axes: dict[str, object] = {}

        self._init_input(dwi_or_dir, subject_ids=subject_ids, max_subjects=max_subjects)
        self._build_ui()
        self._load_subject(0)

    # ------------------------------------------------------------------
    # Input / subject loading
    # ------------------------------------------------------------------

    def _init_input(
        self,
        dwi_or_dir: InputType,
        *,
        subject_ids: Optional[Sequence[str]] = None,
        max_subjects: Optional[int] = None,
    ) -> None:
        """Interpret the input as either a 4-D array or one or more subject roots."""

        if isinstance(dwi_or_dir, np.ndarray):
            self.dwi = np.asarray(dwi_or_dir, dtype=np.float64)
            if self.dwi.ndim != 4:
                raise ValueError("DWI input must be 4-D (nx x ny x nz x nvol).")
            self.root_dirs = []
            self.subjects = [SubjectInfo(path="<array input>", name="array_input")]
            self.review_log_path = self._default_review_log_path()
            self._ensure_review_log_file()
            self._normalize_review_log_file()
            return

        if isinstance(dwi_or_dir, (str, os.PathLike)):
            self.root_dirs = [os.fspath(dwi_or_dir)]
        else:
            self.root_dirs = [os.fspath(p) for p in dwi_or_dir]

        self.subjects = collect_subjects(
            self.root_dirs,
            subject_ids=subject_ids,
            max_subjects=max_subjects,
        )
        self.dwi = load_subject_nifti(self.subjects[0].path)
        self.review_log_path = self._default_review_log_path()
        self._ensure_review_log_file()
        self._normalize_review_log_file()

    def _default_review_log_path(self) -> str:
        """Choose the default JSONL path used to save review decisions."""

        if self.root_dirs:
            base = os.path.abspath(self.root_dirs[0])
            if os.path.isdir(base):
                return os.path.join(base, "diffusion_review_log.jsonl")
        return os.path.abspath("diffusion_review_log.jsonl")

    def _ensure_review_log_file(self) -> None:
        """Create the review log file and its parent directory if needed."""

        assert self.review_log_path is not None
        os.makedirs(os.path.dirname(self.review_log_path) or ".", exist_ok=True)
        if not os.path.exists(self.review_log_path):
            with open(self.review_log_path, "w", encoding="utf-8"):
                pass

    def _normalize_review_log_file(self) -> None:
        """Rewrite the review log into a clean one-line-per-subject format."""

        assert self.review_log_path is not None
        records: dict[str, list[int]] = {}

        with open(self.review_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(payload, dict):
                    continue

                subject_id = payload.get("subject_id")
                if not isinstance(subject_id, str) or not subject_id:
                    continue

                if payload.get("event") not in {None, "bad"}:
                    continue

                raw_volume = payload.get("volume")
                if raw_volume is None:
                    continue

                raw_values = raw_volume if isinstance(raw_volume, list) else [raw_volume]
                subject_volumes = records.setdefault(subject_id, [])
                for raw_value in raw_values:
                    try:
                        volume = int(raw_value)
                    except (TypeError, ValueError):
                        continue
                    if volume not in subject_volumes:
                        subject_volumes.append(volume)

        self._write_review_log(records)

    def _write_review_log(self, records: dict[str, list[int]]) -> None:
        """Persist the normalized bad-volume records back to disk."""

        assert self.review_log_path is not None
        with open(self.review_log_path, "w", encoding="utf-8") as f:
            for subject_id, volumes in records.items():
                payload = {
                    "subject_id": subject_id,
                    "volume": sorted(int(v) for v in volumes),
                }
                f.write(json.dumps(payload) + "\n")

    def _record_bad_volume(self, subject: SubjectInfo, volume: int) -> None:
        """Append a newly marked bad volume for the current subject."""

        assert self.review_log_path is not None
        records: dict[str, list[int]] = {}

        with open(self.review_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue

                subject_id = payload.get("subject_id")
                raw_volume = payload.get("volume")
                if not isinstance(subject_id, str) or raw_volume is None:
                    continue

                raw_values = raw_volume if isinstance(raw_volume, list) else [raw_volume]
                subject_volumes = records.setdefault(subject_id, [])
                for raw_value in raw_values:
                    try:
                        parsed_volume = int(raw_value)
                    except (TypeError, ValueError):
                        continue
                    if parsed_volume not in subject_volumes:
                        subject_volumes.append(parsed_volume)

        subject_volumes = records.setdefault(subject.name, [])
        if volume not in subject_volumes:
            subject_volumes.append(int(volume))

        self._write_review_log(records)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    @staticmethod
    def _shift_axes_left(*axes, delta: float) -> None:
        """Move a group of axes left by a small normalized figure offset."""

        for ax in axes:
            pos = ax.get_position()
            ax.set_position([pos.x0 - delta, pos.y0, pos.width, pos.height])

    def _build_ui(self) -> None:
        """Create the figure, plots, and keyboard hooks for the viewer."""

        self.fig = plt.figure(figsize=(18, 11))
        try:
            self.fig.canvas.manager.set_window_title("dMRI Viewer + Review Log")
        except Exception:
            pass

        self._main_gs = self.fig.add_gridspec(
            2, 3, left=0.03, right=0.98, top=0.94, bottom=0.20, wspace=0.18, hspace=0.25
        )

        self.ax_axial = self.fig.add_subplot(self._main_gs[0, 0])
        self.ax_sagittal = self.fig.add_subplot(self._main_gs[0, 1])
        self.ax_coronal = self.fig.add_subplot(self._main_gs[0, 2])
        self.ax_fft = self.fig.add_subplot(self._main_gs[1, 0])
        self.ax_radial = self.fig.add_subplot(self._main_gs[1, 1])
        self.ax_info = self.fig.add_subplot(self._main_gs[1, 2])

        for ax in (self.ax_axial, self.ax_sagittal, self.ax_coronal, self.ax_fft):
            ax.set_xticks([])
            ax.set_yticks([])

        self.ax_radial.grid(True, alpha=0.3)
        self.ax_info.axis("off")

        self.im_axial = self.ax_axial.imshow(np.zeros((2, 2)), cmap="gray", origin="upper")
        self.ax_axial.set_title("Axial")
        cbar_axial = self.fig.colorbar(self.im_axial, ax=self.ax_axial, fraction=0.046, pad=0.04)

        self.im_sagittal = self.ax_sagittal.imshow(np.zeros((2, 2)), cmap="gray", origin="upper")
        self.ax_sagittal.set_title("Sagittal")
        cbar_sagittal = self.fig.colorbar(self.im_sagittal, ax=self.ax_sagittal, fraction=0.046, pad=0.04)

        self.im_coronal = self.ax_coronal.imshow(np.zeros((2, 2)), cmap="gray", origin="upper")
        self.ax_coronal.set_title("Coronal")
        cbar_coronal = self.fig.colorbar(self.im_coronal, ax=self.ax_coronal, fraction=0.046, pad=0.04)

        self._shift_axes_left(
            self.ax_axial,
            cbar_axial.ax,
            self.ax_sagittal,
            cbar_sagittal.ax,
            self.ax_coronal,
            cbar_coronal.ax,
            delta=0.015,
        )

        self.im_fft = self.ax_fft.imshow(np.zeros((2, 2)), cmap="gray", origin="upper")
        self.ax_fft.set_title("log(|FFT|) (axial slice)")
        self.fig.colorbar(self.im_fft, ax=self.ax_fft, fraction=0.046, pad=0.04)

        (self.line_slice,) = self.ax_radial.plot([], [], lw=1.5, label="Slice")
        (self.line_volume,) = self.ax_radial.plot([], [], lw=1.5, ls="--", label="Volume")
        self.ax_radial.set_xlabel("Frequency (cycles/pixel)")
        self.ax_radial.set_ylabel("Power (a.u.)")
        self.ax_radial.set_title("Radial Power Spectrum")
        self.ax_radial.legend(loc="upper right")
        self.ax_radial.set_yscale("log")

        self.info_text = self.ax_info.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )

        self._build_controls()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)

    def _add_ctrl_axes(self, name: str) -> object:
        """Add a figure-level axes for a control widget and register it."""

        pos = _CTRL_BASE_POS[name]
        ax = self.fig.add_axes(pos)
        self._ctrl_axes[name] = ax
        return ax

    def _build_controls(self) -> None:
        """Create the sliders, buttons, and checkboxes below the plots."""

        self.window_check = CheckButtons(self._add_ctrl_axes("window"), ["Window"], [True])
        self.fixed_wl_check = CheckButtons(self._add_ctrl_axes("fixed_wl"), ["Fix WL"], [False])

        subject_max = max(1, len(self.subjects))
        self.subject_slider = Slider(
            self._add_ctrl_axes("subject"),
            "Subject",
            1,
            subject_max,
            valinit=1,
            valstep=1,
        )

        self.good_button = Button(self._add_ctrl_axes("good"), "Good")
        self.bad_button = Button(self._add_ctrl_axes("bad"), "Bad")

        self.vol_slider = Slider(
            self._add_ctrl_axes("vol"), "Volume", 1, 2, valinit=1, valstep=1
        )
        self.axial_slider = Slider(
            self._add_ctrl_axes("axial"), "Axial", 1, 2, valinit=1, valstep=1
        )
        self.sagittal_slider = Slider(
            self._add_ctrl_axes("sagittal"), "Sagittal", 1, 2, valinit=1, valstep=1
        )
        self.coronal_slider = Slider(
            self._add_ctrl_axes("coronal"), "Coronal", 1, 2, valinit=1, valstep=1
        )

        # Raw min / max text inputs (below the info panel).  Press Enter to apply.
        self.raw_min_box = TextBox(self._add_ctrl_axes("raw_min_box"), "Raw min", initial="0.0")
        self.raw_max_box = TextBox(self._add_ctrl_axes("raw_max_box"), "Raw max", initial="1.0")

        self.subject_slider.on_changed(self._on_subject_slider)
        self.vol_slider.on_changed(self._on_volume_slider)
        self.axial_slider.on_changed(self._on_axial_slider)
        self.sagittal_slider.on_changed(self._on_sagittal_slider)
        self.coronal_slider.on_changed(self._on_coronal_slider)
        self.raw_min_box.on_submit(self._on_raw_min_submit)
        self.raw_max_box.on_submit(self._on_raw_max_submit)
        self.good_button.on_clicked(lambda _evt: self._review_and_advance("good"))
        self.bad_button.on_clicked(lambda _evt: self._review_and_advance("bad"))
        self.window_check.on_clicked(lambda _label: self.update_display())
        self.fixed_wl_check.on_clicked(lambda _label: self.update_display())

    # ------------------------------------------------------------------
    # Fullscreen / resize support
    # ------------------------------------------------------------------

    def _on_resize(self, event=None) -> None:  # noqa: ANN001
        """Reposition control widgets relative to a fixed pixel budget."""

        fig_h_px = self.fig.get_size_inches()[1] * self.fig.dpi
        # Keep ~200 px for the control area regardless of screen size.
        target_frac = float(np.clip(200.0 / max(fig_h_px, 1.0), 0.14, 0.30))
        scale = target_frac / _CTRL_BASELINE_H

        for name, ax in self._ctrl_axes.items():
            base = _CTRL_BASE_POS[name]
            ax.set_position([base[0], base[1] * scale, base[2], base[3] * scale])

        self._main_gs.update(bottom=target_frac + 0.01)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Subject / data management
    # ------------------------------------------------------------------

    def _current_subject(self) -> SubjectInfo:
        """Return the currently selected subject metadata."""

        return self.subjects[self.current_subject_idx]

    def _load_subject(self, idx: int) -> None:
        """Load a subject volume set and refresh slice limits and plots."""

        idx = int(np.clip(idx, 0, len(self.subjects) - 1))
        if idx == self.current_subject_idx and self.dwi is not None:
            self.update_display()
            return

        self.current_subject_idx = idx
        self._consecutive_good_count = 0  # reset streak on explicit subject change
        subj = self._current_subject()
        self.dwi = load_subject_nifti(subj.path)

        nx, ny, nz, nv = self.dwi.shape
        self.sagittal_slice = int(np.clip(self.sagittal_slice if self.sagittal_slice else nx // 2, 1, nx))
        self.coronal_slice = int(np.clip(self.coronal_slice if self.coronal_slice else ny // 2, 1, ny))
        self.axial_slice = int(np.clip(self.axial_slice if self.axial_slice else nz // 2, 1, nz))
        self.vol = int(np.clip(self.vol, 1, nv))

        self._sync_controls_to_current_volume()
        self.update_display()

    def _sync_controls_to_current_volume(self) -> None:
        """Update slider ranges and values to match the loaded subject."""

        assert self.dwi is not None
        nx, ny, nz, nv = self.dwi.shape

        self._updating_controls = True
        try:
            self._set_slider_range(self.vol_slider, 1, max(1, nv), self.vol)
            self._set_slider_range(self.axial_slider, 1, max(1, nz), self.axial_slice)
            self._set_slider_range(self.sagittal_slider, 1, max(1, nx), self.sagittal_slice)
            self._set_slider_range(self.coronal_slider, 1, max(1, ny), self.coronal_slice)

            self._suppress_subject_callback = True
            self._set_slider_range(self.subject_slider, 1, max(1, len(self.subjects)), self.current_subject_idx + 1)

            # Update raw min/max text boxes based on actual data values.
            finite_vals = self.dwi[np.isfinite(self.dwi)]
            if len(finite_vals) > 0:
                d_min = float(finite_vals.min())
                d_max = float(finite_vals.max())
            else:
                d_min, d_max = 0.0, 1.0
            if np.isclose(d_min, d_max):
                d_max = d_min + 1.0
            self._raw_min_val = d_min
            self._raw_max_val = d_max
            self.raw_min_box.set_val(f"{d_min:.4g}")
            self.raw_max_box.set_val(f"{d_max:.4g}")
        finally:
            self._suppress_subject_callback = False
            self._updating_controls = False

    @staticmethod
    def _set_slider_range(slider: Slider, vmin: int, vmax: int, value: int) -> None:
        """Reset a slider range and clamp its current value into bounds."""

        slider.valmin = float(vmin)
        slider.valmax = float(vmax)
        slider.ax.set_xlim(vmin, vmax)
        slider.valstep = 1
        slider.set_val(int(np.clip(value, vmin, vmax)))

    # ------------------------------------------------------------------
    # Volume / subject advancement
    # ------------------------------------------------------------------

    def _advance_volume(self) -> None:
        """Move to the next diffusion volume when one is available."""

        if self.dwi is None:
            return

        nv = self.dwi.shape[-1]
        if self.vol < nv:
            self.vol += 1
        self.update_display()

    def _advance_subject(self) -> None:
        """Move to the next subject (wrapping cyclically)."""

        next_idx = (self.current_subject_idx + 1) % len(self.subjects)
        self._load_subject(next_idx)

    # ------------------------------------------------------------------
    # Slider / button callbacks
    # ------------------------------------------------------------------

    def _on_subject_slider(self, value: float) -> None:
        """Handle subject slider changes from the UI."""

        if self._suppress_subject_callback or self._updating_controls:
            return
        idx = int(round(value)) - 1
        idx = int(np.clip(idx, 0, len(self.subjects) - 1))
        if idx == self.current_subject_idx:
            return
        self._load_subject(idx)

    def _on_volume_slider(self, value: float) -> None:
        """Handle volume slider changes from the UI."""

        if self._updating_controls:
            return
        self.vol = int(round(value))
        self.update_display()

    def _on_axial_slider(self, value: float) -> None:
        """Handle axial slice slider changes from the UI."""

        if self._updating_controls:
            return
        self.axial_slice = int(round(value))
        self.update_display()

    def _on_sagittal_slider(self, value: float) -> None:
        """Handle sagittal slice slider changes from the UI."""

        if self._updating_controls:
            return
        self.sagittal_slice = int(round(value))
        self.update_display()

    def _on_coronal_slider(self, value: float) -> None:
        """Handle coronal slice slider changes from the UI."""

        if self._updating_controls:
            return
        self.coronal_slice = int(round(value))
        self.update_display()

    def _on_raw_min_submit(self, text: str) -> None:
        """Parse and store the raw-min value entered by the user."""

        try:
            self._raw_min_val = float(text)
        except ValueError:
            self.raw_min_box.set_val(f"{self._raw_min_val:.4g}")
            return
        self.update_display()

    def _on_raw_max_submit(self, text: str) -> None:
        """Parse and store the raw-max value entered by the user."""

        try:
            self._raw_max_val = float(text)
        except ValueError:
            self.raw_max_box.set_val(f"{self._raw_max_val:.4g}")
            return
        self.update_display()

    def _review_and_advance(self, decision: str) -> None:
        """Record a review decision then step forward.

        For **bad** votes the volume is recorded and we advance to the next
        volume.  The consecutive-good streak is reset.

        For **good** votes we increment the streak counter.  After
        ``_GOOD_STREAK_THRESHOLD`` consecutive good votes for the same subject
        we automatically advance to the next subject (and reset the counter).
        Otherwise we simply advance to the next volume.
        """

        if self.dwi is None:
            return

        subj = self._current_subject()

        if decision == "bad":
            self._consecutive_good_count = 0
            self._record_bad_volume(subj, self.vol)
            self._advance_volume()
        else:  # "good"
            self._consecutive_good_count += 1
            if self._consecutive_good_count >= _GOOD_STREAK_THRESHOLD:
                self._consecutive_good_count = 0
                self._advance_subject()
            else:
                self._advance_volume()

    def _on_key_press(self, event) -> None:  # noqa: ANN001
        """Support keyboard shortcuts for review and subject navigation."""

        if event.key in {"g", "G"}:
            self._review_and_advance("good")
        elif event.key in {"b", "B"}:
            self._review_and_advance("bad")
        elif event.key in {"right", "d"}:
            self._load_subject((self.current_subject_idx + 1) % len(self.subjects))
        elif event.key in {"left", "a"}:
            self._load_subject((self.current_subject_idx - 1) % len(self.subjects))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _fixed_wl_on(self) -> bool:
        """Return True when the user has enabled manual window-level control."""

        return bool(self.fixed_wl_check.get_status()[0])

    def update_display(self) -> None:
        """Refresh the orthogonal views, FFT, radial plot, and info panel."""

        if self.dwi is None:
            return

        nx, ny, nz, nv = self.dwi.shape

        self.vol = int(np.clip(self.vol, 1, nv))
        self.axial_slice = int(np.clip(self.axial_slice, 1, nz))
        self.sagittal_slice = int(np.clip(self.sagittal_slice, 1, nx))
        self.coronal_slice = int(np.clip(self.coronal_slice, 1, ny))

        self._updating_controls = True
        try:
            self.vol_slider.set_val(self.vol)
            self.axial_slider.set_val(self.axial_slice)
            self.sagittal_slider.set_val(self.sagittal_slice)
            self.coronal_slider.set_val(self.coronal_slice)

            self._suppress_subject_callback = True
            self.subject_slider.set_val(self.current_subject_idx + 1)
        finally:
            self._suppress_subject_callback = False
            self._updating_controls = False

        use_window = self.window_check.get_status()[0]
        fixed_wl = self._fixed_wl_on()

        axial_raw = self.dwi[:, :, self.axial_slice - 1, self.vol - 1]
        sagittal_raw = self.dwi[self.sagittal_slice - 1, :, :, self.vol - 1]
        coronal_raw = self.dwi[:, self.coronal_slice - 1, :, self.vol - 1]

        axial_img = _rot_display(axial_raw)
        sagittal_img = _rot_display(sagittal_raw)
        coronal_img = _rot_display(coronal_raw)

        self.im_axial.set_data(axial_img)
        self.im_sagittal.set_data(sagittal_img)
        self.im_coronal.set_data(coronal_img)

        if fixed_wl:
            # Use the manually entered window level for all three views.
            vmin = self._raw_min_val
            vmax = self._raw_max_val
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
            for im in (self.im_axial, self.im_sagittal, self.im_coronal):
                im.set_clim(vmin, vmax)
        else:
            # Auto-scale each view independently.
            for im, img in [
                (self.im_axial, axial_img),
                (self.im_sagittal, sagittal_img),
                (self.im_coronal, coronal_img),
            ]:
                finite = np.isfinite(img)
                if np.any(finite):
                    vmin = float(np.nanmin(img[finite]))
                    vmax = float(np.nanmax(img[finite]))
                    if np.isclose(vmin, vmax):
                        vmax = vmin + 1.0
                    im.set_clim(vmin, vmax)

        self.ax_axial.set_title(f"Axial | subj {self.current_subject_idx + 1} vol {self.vol} slice {self.axial_slice}")
        self.ax_sagittal.set_title(f"Sagittal | slice {self.sagittal_slice}")
        self.ax_coronal.set_title(f"Coronal | slice {self.coronal_slice}")

        # FFT and radial power are computed from the axial slice only.
        fft_img = axial_img.copy()
        if use_window:
            fft_input = fft_img * _hann2d(*fft_img.shape)
        else:
            fft_input = fft_img

        F = np.fft.fftshift(np.fft.fft2(fft_input))
        mag = np.log(np.abs(F) + 1.0)
        self.im_fft.set_data(mag)

        finite = np.isfinite(mag)
        if np.any(finite):
            vmin = float(np.nanmin(mag[finite]))
            vmax = float(np.nanmax(mag[finite]))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
            self.im_fft.set_clim(vmin, vmax)

        self.ax_fft.set_title("log(|FFT|) (axial slice)")

        power_slice = np.abs(F) ** 2
        f_centers, radial_slice = _radial_power_spectrum(power_slice)

        power_sum = np.zeros_like(power_slice, dtype=np.float64)
        for s in range(nz):
            sl_raw = self.dwi[:, :, s, self.vol - 1]
            sl_img = _rot_display(sl_raw)
            if use_window:
                sl_img = sl_img * _hann2d(*sl_img.shape)
            Fsl = np.fft.fftshift(np.fft.fft2(sl_img))
            power_sum += np.abs(Fsl) ** 2

        _, radial_vol = _radial_power_spectrum(power_sum)

        self.line_slice.set_data(f_centers, radial_slice)
        self.line_volume.set_data(f_centers, radial_vol)
        self.ax_radial.set_xlim(0.0, 0.5)
        self.ax_radial.relim()
        self.ax_radial.autoscale_view(scalex=False, scaley=True)
        self.ax_radial.set_yscale("log")
        self.ax_radial.set_title(f"Radial power | subj {self.current_subject_idx + 1} vol {self.vol} slice {self.axial_slice}")

        streak_msg = f"Good streak: {self._consecutive_good_count}/{_GOOD_STREAK_THRESHOLD}"
        subj = self._current_subject()
        self.info_text.set_text(
            f"Subject: {self.current_subject_idx + 1}/{len(self.subjects)}\n"
            f"ID: {subj.name}\n"
            f"Path: {subj.path}\n"
            f"Volume: {self.vol}/{nv}\n"
            f"Axial: {self.axial_slice}/{nz}\n"
            f"Sagittal: {self.sagittal_slice}/{nx}\n"
            f"Coronal: {self.coronal_slice}/{ny}\n"
            f"Window: {'on' if use_window else 'off'}\n"
            f"Fix WL: {'on' if fixed_wl else 'off'}\n"
            f"{streak_msg}\n"
            f"Log: {self.review_log_path}"
        )

        self.fig.canvas.draw_idle()


def diffusion_image_viewer(
    dwi_or_dir: InputType,
    *,
    subject_ids: Optional[Sequence[str]] = None,
    max_subjects: Optional[int] = None,
    show: bool = True,
) -> DiffusionImageViewer:
    """Create the viewer and optionally start the Matplotlib event loop.

    Parameters
    ----------
    dwi_or_dir:
        A 4-D numpy array, a subject/root directory path, or a sequence of
        root directory paths.
    subject_ids:
        Optional list of subject folder names to include.
    max_subjects:
        Limit scanning to this many subjects when *subject_ids* is not given.
    show:
        When ``True`` (default) ``plt.show()`` is called to open the window.
    """
    viewer = DiffusionImageViewer(dwi_or_dir, subject_ids=subject_ids, max_subjects=max_subjects)
    if show:
        plt.show()
    return viewer


# Alias that matches the MATLAB name more closely.
Diffusion_image_viewer = diffusion_image_viewer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="dMRI orthogonal-view, FFT, and review viewer")
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Root directory or subject directory. Numeric 4-D arrays are "
             "supported via the function call, not the CLI.",
    )
    parser.add_argument(
        "--subject-ids",
        nargs="+",
        metavar="ID",
        default=None,
        help="Only load subjects whose folder name matches one of these IDs.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of subjects loaded when --subject-ids is not given.",
    )
    args = parser.parse_args()

    if args.input is None:
        raise SystemExit("Please pass a root directory or subject directory.")

    diffusion_image_viewer(
        args.input,
        subject_ids=args.subject_ids,
        max_subjects=args.max_subjects,
    )
