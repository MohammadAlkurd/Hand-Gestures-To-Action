from PyQt6.QtWidgets import (QDialog, QFormLayout, QLineEdit, QComboBox, QPushButton,
                             QHBoxLayout, QDoubleSpinBox, QFileDialog, QListWidget,
                             QDialogButtonBox)

from src.keys import AVAILABLE_KEY_NAMES


class KeyReferenceDialog(QDialog):
    """Read-only list of key names the user can type into a keybinding field."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Available Key Names")
        layout = QHBoxLayout(self)
        listw = QListWidget()
        listw.addItems(AVAILABLE_KEY_NAMES)
        layout.addWidget(listw)


class GestureDialog(QDialog):
    """Add/edit dialog for a single gesture binding."""

    def __init__(self, gesture: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gesture")
        self.gesture = dict(gesture) if gesture else {}

        form = QFormLayout(self)

        self.name_edit = QLineEdit(self.gesture.get("name", ""))
        if self.gesture.get("name"):
            self.name_edit.setReadOnly(True)  # name comes from a trained gesture, not freely typed
        form.addRow("Gesture name:", self.name_edit)

        self.action_type = QComboBox()
        self.action_type.addItems(["script", "keypress"])
        self.action_type.setCurrentText(self.gesture.get("action_type", "script"))
        form.addRow("Action type:", self.action_type)

        action_row = QHBoxLayout()
        self.action_value = QLineEdit(self.gesture.get("action_value", ""))
        self.action_value.setPlaceholderText("/path/to/script or 'ctrl+space'")
        action_row.addWidget(self.action_value)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_script)
        action_row.addWidget(browse_btn)
        keys_btn = QPushButton("Key names…")
        keys_btn.clicked.connect(lambda: KeyReferenceDialog(self).exec())
        action_row.addWidget(keys_btn)
        form.addRow("Action value:", action_row)

        self.key_mode = QComboBox()
        self.key_mode.addItems(["normal", "pulse"])
        self.key_mode.setCurrentText(self.gesture.get("key_mode", "normal"))
        form.addRow("Key mode (keypress only):", self.key_mode)

        self.presses_per_second = QDoubleSpinBox()
        self.presses_per_second.setRange(0.5, 50)
        self.presses_per_second.setSingleStep(0.5)
        self.presses_per_second.setValue(self.gesture.get("presses_per_second", 2))
        form.addRow("Presses/second (pulse only):", self.presses_per_second)

        self.cooldown = QDoubleSpinBox()
        self.cooldown.setRange(0, 60)
        self.cooldown.setSingleStep(0.5)
        self.cooldown.setValue(self.gesture.get("cooldown", 2))
        form.addRow("Cooldown after release (seconds):", self.cooldown)

        self.start_after = QDoubleSpinBox()
        self.start_after.setRange(0, 10)
        self.start_after.setSingleStep(0.1)
        self.start_after.setValue(self.gesture.get("start_after", 0))
        form.addRow("Start after held for (seconds):", self.start_after)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                    QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _browse_script(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select script or executable")
        if path:
            self.action_value.setText(path)

    def result_gesture(self) -> dict:
        return {
            "name": self.name_edit.text().strip(),
            "action_type": self.action_type.currentText(),
            "action_value": self.action_value.text().strip(),
            "key_mode": self.key_mode.currentText(),
            "presses_per_second": self.presses_per_second.value(),
            "cooldown": self.cooldown.value(),
            "start_after": self.start_after.value(),
        }
