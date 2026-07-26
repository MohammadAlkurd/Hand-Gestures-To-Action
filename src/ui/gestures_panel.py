from PyQt6.QtWidgets import (QListWidget, QListWidgetItem, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QMessageBox,
                             QInputDialog)

from pathlib import Path

from src.gesture_store import load_gestures, save_gestures, DEFAULT_GESTURE, reset_all
from src.recorder import delete_gesture_cache, delete_all_cache, CACHE_ROOT
from src.ui.gesture_dialog import GestureDialog
from config import settings


class GesturesPanel(QWidget):
    """Left panel: list of configured gestures plus add/edit/remove controls."""

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.gestures = load_gestures()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        self._refresh_list()

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self.add_gesture)
        btn_row.addWidget(btn_add)

        btn_edit = QPushButton("Edit")
        btn_edit.clicked.connect(self.edit_gesture)
        btn_row.addWidget(btn_edit)

        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self.remove_gesture)
        btn_row.addWidget(btn_remove)
        layout.addLayout(btn_row)

        btn_reset = QPushButton("Reset everything (fresh start)")
        btn_reset.clicked.connect(self.reset_everything)
        layout.addWidget(btn_reset)

    def reload(self):
        """Reloads gestures from disk (e.g. after a name was auto-added via recording)."""
        self.gestures = load_gestures()
        self._refresh_list()

    def _refresh_list(self):
        self.list_widget.clear()
        for g in self.gestures:
            text = f"{g['name']}  →  {g['action_type']}:{g['action_value']}"
            if g["action_type"] == "keypress":
                text += f" ({g['key_mode']})"
            self.list_widget.addItem(QListWidgetItem(text))

    def add_gesture(self):
        bound = {g["name"] for g in self.gestures}
        trained = sorted(d.name for d in CACHE_ROOT.iterdir() if d.is_dir()) if CACHE_ROOT.exists() else []
        available = [n for n in trained if n not in bound]
        if not available:
            QMessageBox.information(self, "No trained gestures",
                                     "Record or import a video for a new gesture first (right panel), "
                                     "then add its action binding here.")
            return
        name, ok = QInputDialog.getItem(self, "Add gesture", "Trained gesture:", available, editable=False)
        if not ok:
            return
        preset = dict(DEFAULT_GESTURE)
        preset["name"] = name
        dialog = GestureDialog(preset, parent=self)
        if dialog.exec():
            gesture = dialog.result_gesture()
            if not gesture["name"]:
                QMessageBox.warning(self, "Invalid gesture", "Gesture name is required.")
                return
            self.gestures.append(gesture)
            save_gestures(self.gestures)
            self._refresh_list()

    def edit_gesture(self):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        dialog = GestureDialog(self.gestures[row], parent=self)
        if dialog.exec():
            self.gestures[row] = dialog.result_gesture()
            save_gestures(self.gestures)
            self._refresh_list()

    def reset_everything(self):
        reply = QMessageBox.question(
            self, "Reset everything",
            "This deletes ALL gestures, cached recordings and the trained model. "
            "This cannot be undone. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        reset_all()
        delete_all_cache()
        checkpoint = Path(settings['model']['checkpoint_path'])
        if checkpoint.exists():
            checkpoint.unlink()
        self.reload()
        if self.parent and hasattr(self.parent, "reset_model"):
            self.parent.reset_model()

    def remove_gesture(self):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        name = self.gestures[row]["name"]
        del self.gestures[row]
        save_gestures(self.gestures)
        self._refresh_list()
        delete_gesture_cache(name)
        if self.parent and hasattr(self.parent, "_retrain"):
            self.parent._retrain()
