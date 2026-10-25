"""Settings view: implemented preferences and actual configuration.

The page exposes one persisted preference with real behavior: remembering
the main window's size and position. Read-only sections show the actual
storage location, theme, default preprocessing configuration, and
application identity. Unsupported preferences are explicitly described as
not implemented instead of being presented as working controls.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from associative_memory.presentation.theme.spacing import (
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
    SPACE_XXS,
)
from associative_memory.presentation.viewmodels.settings_viewmodel import (
    SettingsViewModel,
)


class SettingsView(QWidget):
    """The Settings page."""

    def __init__(
        self,
        view_model: SettingsViewModel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        # Preserve the placeholder page's former positional-parent usage.
        if isinstance(view_model, QWidget):
            if parent is not None:
                raise TypeError("parent was provided twice.")
            parent = view_model
            view_model = None

        super().__init__(parent)

        self._view_model = (
            view_model
            if view_model is not None
            else SettingsViewModel(parent=self)
        )

        self._build_layout()
        self._view_model.changed.connect(self._refresh)
        self._refresh()

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(SPACE_SM)

        intro = QLabel(
            "Only settings with implemented behavior are editable here. "
            "The remaining values describe the running configuration.",
            content,
        )
        intro.setObjectName("PageSubtitle")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(self._build_window_section(content))
        layout.addWidget(self._build_configuration_section(content))
        layout.addWidget(self._build_about_section(content))
        layout.addStretch(1)

        scroll.setWidget(content)
        root.addWidget(scroll)

    def _section_card(
        self,
        title: str,
        description: str,
        parent: QWidget,
    ) -> tuple[QFrame, QVBoxLayout]:
        """Build one bordered section and return its body layout."""

        card = QFrame(parent)
        card.setObjectName("ConfigPanel")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        layout.setSpacing(SPACE_XS)

        title_label = QLabel(title, card)
        title_label.setObjectName("SectionTitle")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        description_label = QLabel(description, card)
        description_label.setObjectName("Caption")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        return card, layout

    def _build_window_section(self, parent: QWidget) -> QFrame:
        card, layout = self._section_card(
            "Window",
            "The only editable preference in this build.",
            parent,
        )

        self._remember_geometry_checkbox = QCheckBox(
            "Remember window size and position", card
        )
        self._remember_geometry_checkbox.setToolTip(
            "When enabled, the main window restores its saved size and "
            "position on the next launch."
        )
        self._remember_geometry_checkbox.setAccessibleDescription(
            "Persist and restore the main window size and position."
        )
        self._remember_geometry_checkbox.toggled.connect(
            self._on_remember_geometry_toggled
        )
        layout.addWidget(self._remember_geometry_checkbox)

        self._geometry_hint = QLabel(card)
        self._geometry_hint.setObjectName("Caption")
        self._geometry_hint.setWordWrap(True)
        layout.addWidget(self._geometry_hint)

        actions = QHBoxLayout()
        actions.setSpacing(SPACE_XS)

        self._reset_window_button = QPushButton("Reset Window Layout", card)
        self._reset_window_button.setToolTip(
            "Return the main window to its default size and centered "
            "position. Any saved geometry is discarded."
        )
        self._reset_window_button.clicked.connect(self._on_reset_window_clicked)
        actions.addWidget(self._reset_window_button)
        actions.addStretch(1)

        layout.addLayout(actions)
        return card

    def _build_configuration_section(self, parent: QWidget) -> QFrame:
        card, layout = self._section_card(
            "Current configuration",
            "Read-only values from the running application. Change "
            "preprocessing settings on the Image Dataset page.",
            parent,
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(SPACE_XS)
        grid.setVerticalSpacing(SPACE_XXS)
        grid.setColumnStretch(1, 1)

        grid.addWidget(self._field_label("Experiment storage", card), 0, 0)
        self._storage_value = QLabel(card)
        self._storage_value.setObjectName("HistoryStorageLabel")
        self._storage_value.setWordWrap(True)
        self._storage_value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        grid.addWidget(self._storage_value, 0, 1)

        grid.addWidget(self._field_label("Theme", card), 1, 0)
        self._theme_value = QLabel(card)
        self._theme_value.setObjectName("Caption")
        self._theme_value.setWordWrap(True)
        grid.addWidget(self._theme_value, 1, 1)

        grid.addWidget(
            self._field_label("Default preprocessing (new sessions)", card),
            2,
            0,
        )
        self._preprocessing_value = QLabel(card)
        self._preprocessing_value.setObjectName("HistoryDetails")
        self._preprocessing_value.setWordWrap(True)
        self._preprocessing_value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        grid.addWidget(self._preprocessing_value, 2, 1)

        layout.addLayout(grid)
        return card

    def _build_about_section(self, parent: QWidget) -> QFrame:
        card, layout = self._section_card(
            "About",
            "Application identity and preference scope.",
            parent,
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(SPACE_XS)
        grid.setVerticalSpacing(SPACE_XXS)
        grid.setColumnStretch(1, 1)

        grid.addWidget(self._field_label("Application", card), 0, 0)
        self._application_value = QLabel(card)
        self._application_value.setObjectName("Caption")
        grid.addWidget(self._application_value, 0, 1)

        grid.addWidget(self._field_label("Version", card), 1, 0)
        self._version_value = QLabel(card)
        self._version_value.setObjectName("Caption")
        grid.addWidget(self._version_value, 1, 1)

        layout.addLayout(grid)

        self._unsupported_note = QLabel(card)
        self._unsupported_note.setObjectName("Caption")
        self._unsupported_note.setWordWrap(True)
        self._unsupported_note.setText(
            "Not implemented yet: theme selection, configurable experiment "
            "storage, and startup-page preferences. They are intentionally "
            "not shown as editable settings. The window preference above is "
            "stored with Qt's QSettings mechanism."
        )
        layout.addWidget(self._unsupported_note)
        return card

    @staticmethod
    def _field_label(text: str, parent: QWidget) -> QLabel:
        """Build a consistent field label for read-only values."""

        label = QLabel(text, parent)
        label.setObjectName("SubsectionLabel")
        label.setWordWrap(True)
        return label

    # --------------------------------------------------------------- rendering

    def _refresh(self) -> None:
        """Synchronize controls and read-only values with the view model."""

        view_model = self._view_model

        self._remember_geometry_checkbox.blockSignals(True)
        self._remember_geometry_checkbox.setChecked(
            view_model.remember_window_geometry
        )
        self._remember_geometry_checkbox.blockSignals(False)

        if view_model.remember_window_geometry:
            hint = (
                "The window's size and position are saved when it closes and "
                "restored on the next launch. Disabling this option discards "
                "the saved geometry."
            )
        else:
            hint = (
                "The window opens at its default size on each launch. Enable "
                "this option to save the window's size and position when it "
                "closes and restore them next time."
            )
        self._geometry_hint.setText(hint)

        self._storage_value.setText(view_model.storage_location_text)
        self._theme_value.setText(
            f"{view_model.theme_name} (the only theme implemented)"
        )
        self._preprocessing_value.setText(
            view_model.preprocessing_defaults_text
        )
        self._application_value.setText(view_model.application_name)
        self._version_value.setText(view_model.application_version)

    # ---------------------------------------------------------------- actions

    def _on_remember_geometry_toggled(self, checked: bool) -> None:
        self._view_model.set_remember_window_geometry(checked)

    def _on_reset_window_clicked(self) -> None:
        self._view_model.request_reset_window_layout()
