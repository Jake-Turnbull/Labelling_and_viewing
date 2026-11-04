"""FSLeyes control panel for ranked diffusion QC."""

from __future__ import annotations

from pathlib import Path

import wx
import fsl.data.image as fslimage
import fsleyes.controls.controlpanel as ctrlpanel
import fsleyes.views.orthopanel as orthopanel

from .dataset import Dataset, Subject, load_dataset, read_b_values
from .export import export_project_csvs
from .review import LABEL_FIELDS, ReviewManager
from .storage import ProjectLock, ProjectStore


class DiffusionQCPanel(ctrlpanel.ControlPanel):
    """A compact dockable reviewer which drives the active OrthoPanel."""

    @staticmethod
    def title() -> str:
        return "Diffusion QC"

    @staticmethod
    def supportedViews():
        return [orthopanel.OrthoPanel]

    @staticmethod
    def defaultLayout():
        return {"location": wx.RIGHT}

    def __init__(self, parent, overlayList, displayCtx, viewPanel):
        super().__init__(parent, overlayList, displayCtx, viewPanel)
        self.SetMinSize((300, 520))
        self.dataset: Dataset | None = None
        self.review: ReviewManager | None = None
        self.store: ProjectStore | None = None
        self.project_lock: ProjectLock | None = None
        self.managed_overlay = None
        self.managed_subject_id: str | None = None
        self._updating = False
        self._bvalues: dict[str, tuple[float, ...]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.subject_csv = wx.FilePickerCtrl(
            self, message="Select subject CSV", wildcard="CSV files (*.csv)|*.csv"
        )
        self.ranking_csv = wx.FilePickerCtrl(
            self, message="Select volume ranking CSV", wildcard="CSV files (*.csv)|*.csv"
        )
        self.subject_root = wx.DirPickerCtrl(self, message="Select subject root")
        self.results_dir = wx.DirPickerCtrl(self, message="Select results directory")
        for label, picker in (
            ("Subjects", self.subject_csv),
            ("Ranking", self.ranking_csv),
            ("Subject root", self.subject_root),
            ("Results", self.results_dir),
        ):
            sizer.Add(wx.StaticText(self, label=label), flag=wx.TOP, border=5)
            sizer.Add(picker, flag=wx.EXPAND)

        self.threshold = wx.SpinCtrl(self, min=1, max=105, initial=5)
        threshold_row = wx.BoxSizer(wx.HORIZONTAL)
        threshold_row.Add(wx.StaticText(self, label="Good-screen threshold"), 1, wx.ALIGN_CENTER_VERTICAL)
        threshold_row.Add(self.threshold, 0, wx.LEFT, 6)
        sizer.Add(threshold_row, flag=wx.EXPAND | wx.TOP, border=8)

        self.load_button = wx.Button(self, label="Load / Resume")
        sizer.Add(self.load_button, flag=wx.EXPAND | wx.TOP, border=8)
        self.export_button = wx.Button(self, label="Export CSVs")
        sizer.Add(self.export_button, flag=wx.EXPAND | wx.TOP, border=5)
        sizer.Add(wx.StaticLine(self), flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=8)

        self.subject_choice = wx.Choice(self)
        self.info = wx.StaticText(self, label="Choose inputs, then load a QC project.")
        self.info.Wrap(280)
        sizer.Add(wx.StaticText(self, label="Subject"), flag=wx.TOP, border=4)
        sizer.Add(self.subject_choice, flag=wx.EXPAND)
        sizer.Add(self.info, flag=wx.EXPAND | wx.TOP, border=8)

        rank_row = wx.BoxSizer(wx.HORIZONTAL)
        rank_row.Add(wx.StaticText(self, label="Rank"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.rank = wx.SpinCtrl(self, min=1, max=1, initial=1)
        rank_row.Add(self.rank, 0, wx.LEFT, 6)
        self.volume_info = wx.StaticText(self, label="Volume index: -\nB-value: -\nLabel: -")
        sizer.Add(rank_row, flag=wx.TOP, border=8)
        sizer.Add(self.volume_info, flag=wx.TOP, border=4)

        navigation = wx.BoxSizer(wx.HORIZONTAL)
        self.previous_button = wx.Button(self, label="Previous")
        self.next_button = wx.Button(self, label="Next")
        navigation.Add(self.previous_button, 1, wx.RIGHT, 4)
        navigation.Add(self.next_button, 1, wx.LEFT, 4)
        sizer.Add(navigation, flag=wx.EXPAND | wx.TOP, border=8)

        decisions = wx.GridSizer(rows=2, cols=2, vgap=4, hgap=4)
        self.decision_buttons = {}
        for label in ("Good", "Bad", "Uncertain", "Skip"):
            button = wx.Button(self, label=label)
            self.decision_buttons[label] = button
            decisions.Add(button, 1, wx.EXPAND)
        sizer.Add(decisions, flag=wx.EXPAND | wx.TOP, border=8)

        sizer.Add(wx.StaticText(self, label="Reviewer note"), flag=wx.TOP, border=8)
        self.notes = wx.TextCtrl(self, style=wx.TE_MULTILINE, size=(-1, 64))
        sizer.Add(self.notes, flag=wx.EXPAND | wx.TOP, border=3)
        self.SetSizer(sizer)

        self.load_button.Bind(wx.EVT_BUTTON, self._on_load)
        self.export_button.Bind(wx.EVT_BUTTON, self._on_export)
        self.subject_choice.Bind(wx.EVT_CHOICE, self._on_subject_changed)
        self.rank.Bind(wx.EVT_SPINCTRL, self._on_rank_changed)
        self.threshold.Bind(wx.EVT_SPINCTRL, self._on_threshold_changed)
        self.previous_button.Bind(wx.EVT_BUTTON, lambda _event: self._step_rank(-1))
        self.next_button.Bind(wx.EVT_BUTTON, lambda _event: self._step_rank(1))
        for label, button in self.decision_buttons.items():
            button.Bind(wx.EVT_BUTTON, lambda _event, choice=label: self._on_decision(choice))
        self.notes.Bind(wx.EVT_KILL_FOCUS, self._on_note_changed)

    def _on_load(self, _event) -> None:
        try:
            if self.project_lock is not None:
                self.project_lock.release()
            self.dataset = load_dataset(
                self.subject_csv.GetPath(),
                self.ranking_csv.GetPath(),
                self.subject_root.GetPath(),
            )
            results_path = Path(self.results_dir.GetPath()).expanduser()
            results_path.mkdir(parents=True, exist_ok=True)
            self.store = ProjectStore(results_path / "diffusion_qc_project.json")
            self.project_lock = ProjectLock(self.store.path)
            self.project_lock.acquire()
            saved_state = self.store.load() if self.store.path.exists() else None
            self.review = ReviewManager(
                self.dataset,
                saved_state,
                good_threshold=self.threshold.GetValue(),
            )
            self.review.state["settings"].update(
                {
                    "subject_root": str(Path(self.subject_root.GetPath()).expanduser()),
                    "subject_csv": str(Path(self.subject_csv.GetPath()).expanduser()),
                    "ranking_csv": str(Path(self.ranking_csv.GetPath()).expanduser()),
                    "results_dir": str(results_path),
                }
            )
            self.threshold.SetValue(int(self.review.state["settings"]["good_threshold"]))
            self.rank.SetRange(1, len(self.dataset.ranking))
            self.subject_choice.SetItems([subject.subject_id for subject in self.dataset.subjects])
            self._bvalues = {
                subject.subject_id: read_b_values(subject.bval_path)
                for subject in self.dataset.subjects
                if subject.loading_error is None
            }
            if self.review.current_subject_id is None:
                self.review.select_subject(self.dataset.subjects[0].subject_id)
            self._persist()
            self._show_current()
        except Exception as exc:
            if self.project_lock is not None:
                self.project_lock.release()
                self.project_lock = None
            wx.MessageBox(str(exc), "Diffusion QC", wx.OK | wx.ICON_ERROR, self)

    def _current_subject(self) -> Subject | None:
        if self.dataset is None or self.review is None:
            return None
        subject_id = self.review.current_subject_id
        return next((subject for subject in self.dataset.subjects if subject.subject_id == subject_id), None)

    def _current_volume(self) -> int | None:
        if self.dataset is None or self.review is None or self.review.current_subject_id is None:
            return None
        rank_index = max(1, min(int(self.review.state["progress"].get("current_rank", 1)), len(self.dataset.ranking)))
        return self.dataset.ranking[rank_index - 1].volume

    def _show_current(self) -> None:
        subject = self._current_subject()
        if subject is None or self.review is None or self.dataset is None:
            self.info.SetLabel("All subjects are complete or marked with loading errors.")
            self.volume_info.SetLabel("Volume index: -\nB-value: -\nLabel: -")
            self._remove_managed_overlay()
            return

        self._updating = True
        try:
            self.subject_choice.SetSelection(
                next(i for i, item in enumerate(self.dataset.subjects) if item.subject_id == subject.subject_id)
            )
            volume = self._current_volume()
            rank = next((entry.rank for entry in self.dataset.ranking if entry.volume == volume), 1)
            self.rank.SetValue(rank)
            if subject.loading_error:
                for button in self.decision_buttons.values():
                    button.Enable(False)
                self.info.SetLabel(f"{subject.subject_id}\nLoading error: {subject.loading_error}")
                self._remove_managed_overlay()
                self.volume_info.SetLabel("Volume index: -\nB-value: -\nLabel: Loading error")
                return

            state = self.review.state["subjects"][subject.subject_id]
            self.info.SetLabel(
                f"{subject.subject_id}\n{subject.year:04d}-{subject.month:02d} | "
                f"Prior group: {subject.group}\nYlin: {subject.ylin:g}\n"
                f"Status: {self.review.subject_status(subject.subject_id)}"
            )
            label = "Not inspected"
            for label_name, field in LABEL_FIELDS.items():
                if volume in state[field]:
                    label = label_name
                    break
            if volume in state["skipped_volumes"]:
                label = "Skipped"
            bvalue = self._bvalues[subject.subject_id][volume]
            self.volume_info.SetLabel(
                f"Rank: {rank} / {len(self.dataset.ranking)}\n"
                f"Volume index: {volume}\nB-value: {bvalue:g}\nLabel: {label}"
            )
            for button in self.decision_buttons.values():
                button.Enable(True)
            if self.notes.GetValue() != state.get("notes", ""):
                self.notes.ChangeValue(state.get("notes", ""))
            self._show_overlay(subject, volume)
        finally:
            self._updating = False

    def _show_overlay(self, subject: Subject, volume: int) -> None:
        try:
            if self.managed_subject_id != subject.subject_id:
                self._remove_managed_overlay()
                overlay = fslimage.Image(str(subject.image_path))
                self.overlayList.append(overlay)
                self.managed_overlay = overlay
                self.managed_subject_id = subject.subject_id
                # self.displayCtx.selectedOverlay = overlay
                self.displayCtx.selectedOverlay = self.overlayList.index(overlay)
            self.displayCtx.getOpts(self.managed_overlay).volume = volume
            if self.review is not None:
                self.review.clear_loading_error(subject.subject_id)
                self._persist()
            for button in self.decision_buttons.values():
                button.Enable(True)
        except Exception as exc:
            if self.review is not None:
                self.review.mark_loading_error(subject.subject_id, str(exc))
                self._persist()
            self._remove_managed_overlay()
            for button in self.decision_buttons.values():
                button.Enable(False)
            self.info.SetLabel(f"{subject.subject_id}\nLoading error: {exc}")
            self.volume_info.SetLabel("Volume index: -\nB-value: -\nLabel: Loading error")

    def _remove_managed_overlay(self) -> None:
        if self.managed_overlay is not None and self.managed_overlay in self.overlayList:
            self.overlayList.remove(self.managed_overlay)
        self.managed_overlay = None
        self.managed_subject_id = None

    def _on_subject_changed(self, _event) -> None:
        if self._updating or self.review is None:
            return
        subject_id = self.subject_choice.GetStringSelection()
        self.review.select_subject(subject_id)
        volume = self.review.next_volume(subject_id)
        if volume is None and self.dataset is not None:
            self.review.select_volume(subject_id, self.dataset.ranking[0].volume)
        self._persist()
        self._show_current()

    def _on_rank_changed(self, _event) -> None:
        if self._updating or self.dataset is None or self.review is None:
            return
        subject_id = self.review.current_subject_id
        if subject_id is not None:
            self.review.select_volume(subject_id, self.dataset.ranking[self.rank.GetValue() - 1].volume)
            self._persist()
            self._show_current()

    def _step_rank(self, step: int) -> None:
        if self.dataset is None:
            return
        self.rank.SetValue(max(1, min(len(self.dataset.ranking), self.rank.GetValue() + step)))
        self._on_rank_changed(None)

    def _on_threshold_changed(self, _event) -> None:
        if self._updating or self.review is None:
            return
        self.review.state["settings"]["good_threshold"] = self.threshold.GetValue()
        self._persist()
        self._show_current()

    def _on_decision(self, label: str) -> None:
        if self.review is None or self.store is None:
            return
        subject_id = self.review.current_subject_id
        volume = self._current_volume()
        if subject_id is None or volume is None:
            return
        if label == "Skip":
            self.review.skip_volume(subject_id, volume)
        else:
            self.review.record_decision(subject_id, volume, label)
        self._persist()
        if self.review.next_volume(subject_id) is None:
            self.review.advance_subject()
        self._show_current()

    def _on_note_changed(self, event) -> None:
        if not self._updating and self.review is not None and self.review.current_subject_id is not None:
            self.review.set_note(self.review.current_subject_id, self.notes.GetValue())
            self._persist()
        event.Skip()

    def _persist(self) -> None:
        if self.store is not None and self.review is not None:
            self.store.save(self.review.state)

    def _on_export(self, _event) -> None:
        if self.dataset is None or self.review is None or self.store is None:
            return
        try:
            volume_path, subject_path = export_project_csvs(
                self.dataset, self.review.state, self.store.path.parent
            )
            wx.MessageBox(
                f"Exported:\n{volume_path}\n{subject_path}",
                "Diffusion QC",
                wx.OK | wx.ICON_INFORMATION,
                self,
            )
        except Exception as exc:
            wx.MessageBox(str(exc), "Diffusion QC export", wx.OK | wx.ICON_ERROR, self)

    def destroy(self):
        self._remove_managed_overlay()
        if self.project_lock is not None:
            self.project_lock.release()
            self.project_lock = None
        super().destroy()