from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider


ArrayLike4D = np.ndarray
PathLike = Union[str, os.PathLike]
InputType = Union[ArrayLike4D, PathLike, Sequence[PathLike]]


@dataclass
class SubjectInfo:
    path: str
    name: str

def _folder_label(path: str) -> str:
    label = os.path.basename(os.path.normpath(path))
    return label if label else path


def _has_ap_file(subj_path: str) -> bool:
    f1 = os.path.join(subj_path, "dMRI", "raw", "AP.nii")
    f2 = os.path.join(subj_path, "dMRI", "raw", "AP.nii.gz")
    return os.path.isfile(f1) or os.path.isfile(f2)


def collect_subjects(root_dirs: Sequence[PathLike]) -> List[SubjectInfo]:
    subjects: List[SubjectInfo] = []

    for root in root_dirs:
        root_dir = os.fspath(root)
        if not os.path.isdir(root_dir):
            continue

        # Case 1: the root itself is a subject folder
        if _has_ap_file(root_dir):
            subjects.append(SubjectInfo(path=root_dir, name=_folder_label(root_dir)))
            continue

        # Case 2: scan immediate child folders
        try:
            entries = [
                e for e in os.scandir(root_dir)
                if e.is_dir() and e.name not in {".", ".."}
            ]
        except FileNotFoundError:
            continue

        for entry in entries:
            subj_path = entry.path
            if _has_ap_file(subj_path):
                subjects.append(SubjectInfo(path=subj_path, name=_folder_label(subj_path)))

    if not subjects:
        raise FileNotFoundError(
            "No subject folders with dMRI/raw/AP.nii or AP.nii.gz were found."
        )

    return subjects


def load_subject_nifti(subj_path: str) -> np.ndarray:
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
    return np.rot90(np.asarray(img2d))


def _hann2d(nx: int, ny: int) -> np.ndarray:
    wx = np.hanning(nx)
    wy = np.hanning(ny)
    return np.outer(wx, wy)


def _radial_power_spectrum(power2d: np.ndarray, nbins: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, dwi_or_dir: InputType):
        self.subjects: List[SubjectInfo] = []
        self.current_subject_idx = -1
        self.dwi: np.ndarray | None = None
        self.root_dirs: List[str] = []
        self.review_log_path: str | None = None
        self._suppress_subject_callback = False
        self._updating_controls = False

        self.vol = 1
        self.axial_slice = 0
        self.sagittal_slice = 0
        self.coronal_slice = 0

        self.fig = None
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

        self.subject_slider = None
        self.vol_slider = None
        self.axial_slider = None
        self.sagittal_slider = None
        self.coronal_slider = None
        self.window_check = None
        self.good_button = None
        self.bad_button = None

        self._init_input(dwi_or_dir)
        self._build_ui()
        self._load_subject(0)

    def _init_input(self, dwi_or_dir: InputType) -> None:
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

        self.subjects = collect_subjects(self.root_dirs)
        self.dwi = load_subject_nifti(self.subjects[0].path)
        self.review_log_path = self._default_review_log_path()
        self._ensure_review_log_file()
        self._normalize_review_log_file()

    def _default_review_log_path(self) -> str:
        if self.root_dirs:
            base = os.path.abspath(self.root_dirs[0])
            if os.path.isdir(base):
                return os.path.join(base, "diffusion_review_log.jsonl")
        return os.path.abspath("diffusion_review_log.jsonl")

    def _ensure_review_log_file(self) -> None:
        assert self.review_log_path is not None
        os.makedirs(os.path.dirname(self.review_log_path) or ".", exist_ok=True)
        if not os.path.exists(self.review_log_path):
            with open(self.review_log_path, "w", encoding="utf-8"):
                pass

    def _normalize_review_log_file(self) -> None:
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
        assert self.review_log_path is not None
        with open(self.review_log_path, "w", encoding="utf-8") as f:
            for subject_id, volumes in records.items():
                payload = {
                    "subject_id": subject_id,
                    "volume": sorted(int(v) for v in volumes),
                }
                f.write(json.dumps(payload) + "\n")

    def _record_bad_volume(self, subject: SubjectInfo, volume: int) -> None:
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

    def _build_ui(self) -> None:
        self.fig = plt.figure(figsize=(18, 11))
        try:
            self.fig.canvas.manager.set_window_title("dMRI Viewer + Review Log")
        except Exception:
            pass

        gs = self.fig.add_gridspec(
            2, 3, left=0.03, right=0.98, top=0.94, bottom=0.20, wspace=0.18, hspace=0.25
        )

        self.ax_axial = self.fig.add_subplot(gs[0, 0])
        self.ax_sagittal = self.fig.add_subplot(gs[0, 1])
        self.ax_coronal = self.fig.add_subplot(gs[0, 2])
        self.ax_fft = self.fig.add_subplot(gs[1, 0])
        self.ax_radial = self.fig.add_subplot(gs[1, 1])
        self.ax_info = self.fig.add_subplot(gs[1, 2])

        for ax in (self.ax_axial, self.ax_sagittal, self.ax_coronal, self.ax_fft):
            ax.set_xticks([])
            ax.set_yticks([])

        self.ax_radial.grid(True, alpha=0.3)
        self.ax_info.axis("off")

        self.im_axial = self.ax_axial.imshow(np.zeros((2, 2)), cmap="gray", origin="upper")
        self.ax_axial.set_title("Axial")
        self.fig.colorbar(self.im_axial, ax=self.ax_axial, fraction=0.046, pad=0.04)

        self.im_sagittal = self.ax_sagittal.imshow(np.zeros((2, 2)), cmap="gray", origin="upper")
        self.ax_sagittal.set_title("Sagittal")
        self.fig.colorbar(self.im_sagittal, ax=self.ax_sagittal, fraction=0.046, pad=0.04)

        self.im_coronal = self.ax_coronal.imshow(np.zeros((2, 2)), cmap="gray", origin="upper")
        self.ax_coronal.set_title("Coronal")
        self.fig.colorbar(self.im_coronal, ax=self.ax_coronal, fraction=0.046, pad=0.04)

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

    def _build_controls(self) -> None:
        self.window_check = CheckButtons(self.fig.add_axes([0.05, 0.07, 0.11, 0.07]), ["Window"], [True])

        subject_max = max(1, len(self.subjects))
        self.subject_slider = Slider(
            self.fig.add_axes([0.20, 0.12, 0.30, 0.025]),
            "Subject",
            1,
            subject_max,
            valinit=1,
            valstep=1,
        )
        self.good_button = Button(self.fig.add_axes([0.53, 0.105, 0.09, 0.04]), "Good")
        self.bad_button = Button(self.fig.add_axes([0.64, 0.105, 0.09, 0.04]), "Bad")

        self.vol_slider = Slider(self.fig.add_axes([0.20, 0.07, 0.30, 0.025]), "Volume", 1, 2, valinit=1, valstep=1)
        slider_x = 0.58
        slider_w = 0.28
        slider_h = 0.018
        self.axial_slider = Slider(self.fig.add_axes([slider_x, 0.070, slider_w, slider_h]), "Axial", 1, 2, valinit=1, valstep=1)
        self.sagittal_slider = Slider(self.fig.add_axes([slider_x, 0.045, slider_w, slider_h]), "Sagittal", 1, 2, valinit=1, valstep=1)
        self.coronal_slider = Slider(self.fig.add_axes([slider_x, 0.020, slider_w, slider_h]), "Coronal", 1, 2, valinit=1, valstep=1)

        self.subject_slider.on_changed(self._on_subject_slider)
        self.vol_slider.on_changed(self._on_volume_slider)
        self.axial_slider.on_changed(self._on_axial_slider)
        self.sagittal_slider.on_changed(self._on_sagittal_slider)
        self.coronal_slider.on_changed(self._on_coronal_slider)
        self.good_button.on_clicked(lambda _evt: self._review_and_advance("good"))
        self.bad_button.on_clicked(lambda _evt: self._review_and_advance("bad"))
        self.window_check.on_clicked(lambda _label: self.update_display())

    def _current_subject(self) -> SubjectInfo:
        return self.subjects[self.current_subject_idx]

    def _load_subject(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, len(self.subjects) - 1))
        if idx == self.current_subject_idx and self.dwi is not None:
            self.update_display()
            return

        self.current_subject_idx = idx
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
        finally:
            self._suppress_subject_callback = False
            self._updating_controls = False

    @staticmethod
    def _set_slider_range(slider: Slider, vmin: int, vmax: int, value: int) -> None:
        slider.valmin = float(vmin)
        slider.valmax = float(vmax)
        slider.ax.set_xlim(vmin, vmax)
        slider.valstep = 1
        slider.set_val(int(np.clip(value, vmin, vmax)))

    def _advance_volume(self) -> None:
        if self.dwi is None:
            return

        nv = self.dwi.shape[-1]
        if self.vol < nv:
            self.vol += 1
        self.update_display()

    def _on_subject_slider(self, value: float) -> None:
        if self._suppress_subject_callback or self._updating_controls:
            return
        idx = int(round(value)) - 1
        idx = int(np.clip(idx, 0, len(self.subjects) - 1))
        if idx == self.current_subject_idx:
            return
        self._load_subject(idx)

    def _on_volume_slider(self, value: float) -> None:
        if self._updating_controls:
            return
        self.vol = int(round(value))
        self.update_display()

    def _on_axial_slider(self, value: float) -> None:
        if self._updating_controls:
            return
        self.axial_slice = int(round(value))
        self.update_display()

    def _on_sagittal_slider(self, value: float) -> None:
        if self._updating_controls:
            return
        self.sagittal_slice = int(round(value))
        self.update_display()

    def _on_coronal_slider(self, value: float) -> None:
        if self._updating_controls:
            return
        self.coronal_slice = int(round(value))
        self.update_display()

    def _review_and_advance(self, decision: str) -> None:
        if self.dwi is None:
            return

        subj = self._current_subject()
        if decision == "bad":
            self._record_bad_volume(subj, self.vol)
        self._advance_volume()

    def _on_key_press(self, event) -> None:
        if event.key in {"g", "G"}:
            self._review_and_advance("good")
        elif event.key in {"b", "B"}:
            self._review_and_advance("bad")
        elif event.key in {"right", "d"}:
            self._load_subject((self.current_subject_idx + 1) % len(self.subjects))
        elif event.key in {"left", "a"}:
            self._load_subject((self.current_subject_idx - 1) % len(self.subjects))

    def update_display(self) -> None:
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

        axial_raw = self.dwi[:, :, self.axial_slice - 1, self.vol - 1]
        sagittal_raw = self.dwi[self.sagittal_slice - 1, :, :, self.vol - 1]
        coronal_raw = self.dwi[:, self.coronal_slice - 1, :, self.vol - 1]

        axial_img = _rot_display(axial_raw)
        sagittal_img = _rot_display(sagittal_raw)
        coronal_img = _rot_display(coronal_raw)

        self.im_axial.set_data(axial_img)
        self.im_sagittal.set_data(sagittal_img)
        self.im_coronal.set_data(coronal_img)

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
            f"Log: {self.review_log_path}"
        )

        self.fig.canvas.draw_idle()


def diffusion_image_viewer(dwi_or_dir: InputType, *, show: bool = True) -> DiffusionImageViewer:
    """MATLAB-style entry point."""
    viewer = DiffusionImageViewer(dwi_or_dir)
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
        help="Pass a root directory or subject directory. Numeric 4-D arrays are supported via the function call, not the CLI.",
    )
    args = parser.parse_args()

    if args.input is None:
        raise SystemExit("Please pass a root directory or subject directory.")

    diffusion_image_viewer(args.input)
