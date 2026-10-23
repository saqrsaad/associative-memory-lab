"""Step navigator for inspectable calculation traces.

The navigator is presentation-only. It renders caller-provided trace steps
and never performs numerical work itself. A step carries a short summary
(shown by default) and an optional exact-values detail that the user can
reveal on demand, so large patterns do not flood the page with numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from associative_memory.presentation.theme.spacing import (
    SPACE_SM,
    SPACE_XS,
    SPACE_XXS,
    SPACE_XXXS,
)

_OBJECT_NAME = "StepNavigator"
_LIST_OBJECT_NAME = "TraceStepList"
_CONTENT_OBJECT_NAME = "TraceContent"
_EQUATION_OBJECT_NAME = "TraceEquation"
_INDEX_OBJECT_NAME = "TraceStepIndex"
_DETAIL_OBJECT_NAME = "TraceDetailToggle"


@dataclass(frozen=True)
class TraceStep:
    """One inspectable step of a calculation trace.

    Attributes:
        title: Short navigation label.
        summary: Default content, intentionally compact for large data.
        detail: Optional exact values revealed by the detail toggle.
        equation: Optional equation displayed above the content.
    """

    title: str
    summary: str
    detail: str = ""
    equation: str = ""


class StepNavigator(QWidget):
    """Navigate a list of :class:`TraceStep` objects.

    Shows the selected step's summary by default. When the step contains a
    non-empty ``detail`` different from its summary, a "Show exact values"
    toggle lets the user inspect the complete values. The trace content is
    selectable and scrollable.
    """

    step_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(_OBJECT_NAME)

        self._steps: tuple[TraceStep, ...] = ()
        self._current_index = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_XXS)

        self._step_list = QListWidget(self)
        self._step_list.setObjectName(_LIST_OBJECT_NAME)
        self._step_list.setMinimumWidth(210)
        self._step_list.setMaximumWidth(280)
        self._step_list.setMinimumHeight(200)
        self._step_list.currentRowChanged.connect(self._on_row_changed)

        content_panel = QWidget(self)
        content_panel.setObjectName("TraceContentPanel")
        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(SPACE_XXXS)

        header = QHBoxLayout()
        header.setSpacing(SPACE_XS)
        self._index_label = QLabel(content_panel)
        self._index_label.setObjectName(_INDEX_OBJECT_NAME)
        header.addWidget(self._index_label)
        header.addStretch(1)
        content_layout.addLayout(header)

        self._equation_label = QLabel(content_panel)
        self._equation_label.setObjectName(_EQUATION_OBJECT_NAME)
        self._equation_label.setWordWrap(True)
        self._equation_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        content_layout.addWidget(self._equation_label)

        self._content_edit = QPlainTextEdit(content_panel)
        self._content_edit.setObjectName(_CONTENT_OBJECT_NAME)
        self._content_edit.setReadOnly(True)
        self._content_edit.setMinimumHeight(180)
        self._content_edit.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth
        )
        content_layout.addWidget(self._content_edit, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(SPACE_XS)

        self._detail_toggle = QCheckBox("Show exact values", content_panel)
        self._detail_toggle.setObjectName(_DETAIL_OBJECT_NAME)
        self._detail_toggle.setToolTip(
            "Reveal the complete numeric values for this step."
        )
        self._detail_toggle.toggled.connect(self._render_current)
        self._detail_toggle.hide()
        controls.addWidget(self._detail_toggle)
        controls.addStretch(1)

        self._previous_button = QPushButton("Previous", content_panel)
        self._previous_button.setObjectName("TracePrevious")
        self._previous_button.setToolTip("Show the previous trace step.")
        self._previous_button.clicked.connect(self._go_previous)
        controls.addWidget(self._previous_button)

        self._next_button = QPushButton("Next", content_panel)
        self._next_button.setObjectName("TraceNext")
        self._next_button.setToolTip("Show the next trace step.")
        self._next_button.clicked.connect(self._go_next)
        controls.addWidget(self._next_button)

        content_layout.addLayout(controls)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(SPACE_SM)
        body.addWidget(self._step_list)
        body.addWidget(content_panel, stretch=1)
        layout.addLayout(body)

        self._render_empty()

    # ------------------------------------------------------------------ properties

    @property
    def steps(self) -> tuple[TraceStep, ...]:
        """The currently loaded trace steps."""

        return self._steps

    @property
    def current_index(self) -> int:
        """Zero-based selected step index, or ``-1`` when no step exists."""

        return self._current_index

    @property
    def current_step(self) -> TraceStep | None:
        """The selected step, or ``None``."""

        if 0 <= self._current_index < len(self._steps):
            return self._steps[self._current_index]
        return None

    @property
    def content_text(self) -> str:
        """The text currently visible in the step content area."""

        return self._content_edit.toPlainText()

    @property
    def show_exact_values(self) -> bool:
        """Whether exact-value detail is currently requested."""

        return self._detail_toggle.isChecked()

    # ------------------------------------------------------------------ content

    def set_steps(self, steps: Sequence[TraceStep]) -> None:
        """Replace the displayed trace steps and select the first one."""

        self._steps = tuple(steps)
        self._step_list.blockSignals(True)
        self._step_list.clear()
        for step in self._steps:
            self._step_list.addItem(QListWidgetItem(step.title))
        self._step_list.blockSignals(False)

        if not self._steps:
            self._current_index = -1
            self._render_empty()
            self._sync_navigation()
            return

        self.set_current_index(0)

    def set_current_index(self, index: int) -> None:
        """Select a step by index, clamped to the available range."""

        if not self._steps:
            self._current_index = -1
            self._render_empty()
            self._sync_navigation()
            return

        bounded = max(0, min(int(index), len(self._steps) - 1))
        self._step_list.blockSignals(True)
        self._step_list.setCurrentRow(bounded)
        self._step_list.blockSignals(False)
        self._current_index = bounded
        self._detail_toggle.blockSignals(True)
        self._detail_toggle.setChecked(False)
        self._detail_toggle.blockSignals(False)
        self._render_current()
        self._sync_navigation()
        self.step_changed.emit(bounded)

    def clear(self) -> None:
        """Remove every trace step."""

        self.set_steps(())

    def set_exact_values_visible(self, visible: bool) -> None:
        """Programmatically show or hide exact-value detail."""

        self._detail_toggle.setChecked(bool(visible))
        self._render_current()

    # ------------------------------------------------------------------ actions

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._steps):
            self._current_index = -1
            self._render_empty()
            self._sync_navigation()
            return
        if row == self._current_index:
            return
        self._current_index = row
        self._detail_toggle.blockSignals(True)
        self._detail_toggle.setChecked(False)
        self._detail_toggle.blockSignals(False)
        self._render_current()
        self._sync_navigation()
        self.step_changed.emit(row)

    def _go_previous(self) -> None:
        self.set_current_index(self._current_index - 1)

    def _go_next(self) -> None:
        self.set_current_index(self._current_index + 1)

    # ------------------------------------------------------------------ rendering

    def _render_current(self) -> None:
        step = self.current_step
        if step is None:
            self._render_empty()
            return

        has_detail = bool(step.detail) and step.detail != step.summary
        self._detail_toggle.setVisible(has_detail)
        self._detail_toggle.setEnabled(has_detail)
        if not has_detail and self._detail_toggle.isChecked():
            self._detail_toggle.blockSignals(True)
            self._detail_toggle.setChecked(False)
            self._detail_toggle.blockSignals(False)

        use_detail = has_detail and self._detail_toggle.isChecked()
        self._content_edit.setPlainText(step.detail if use_detail else step.summary)
        self._index_label.setText(
            f"Step {self._current_index + 1} of {len(self._steps)}"
        )
        self._equation_label.setText(step.equation)
        self._equation_label.setVisible(bool(step.equation))
        self._sync_navigation()

    def _render_empty(self) -> None:
        self._content_edit.setPlainText("No calculation trace available.")
        self._index_label.setText("No trace steps")
        self._equation_label.clear()
        self._equation_label.hide()
        self._detail_toggle.hide()

    def _sync_navigation(self) -> None:
        count = len(self._steps)
        self._previous_button.setEnabled(count > 0 and self._current_index > 0)
        self._next_button.setEnabled(
            count > 0 and self._current_index < count - 1
        )
        self.setAccessibleName("Calculation trace")
        self.setAccessibleDescription(
            f"{count} trace step(s); current step {self._current_index + 1}."
            if count
            else "No calculation trace."
        )
