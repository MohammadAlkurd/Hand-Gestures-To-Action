from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSplitter, QCheckBox, QPushButton, QLineEdit,
                             QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

from src.ui.gestures_panel import GesturesPanel
from src.inference import InferenceWorker
from pathlib import Path
from src.recorder import import_video_file, MAX_RECORD_SECONDS
from src.trainer import train_from_cache
from src.gesture_store import ensure_gesture
from config import settings


class TrainThread(QThread):
    """Runs training off the cached landmarks without freezing the UI."""

    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def run(self):
        try:
            train_from_cache()
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setup_ui()
        self.start_inference()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: gestures list and add/edit/remove controls.
        self.gestures_panel = GesturesPanel(self)
        splitter.addWidget(self.gestures_panel)

        # Right: camera preview + prediction + test mode toggle.
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.camera_label = QLabel("Camera preview")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(400, 300)
        self.camera_label.setStyleSheet("background-color: black; color: white;")
        right_layout.addWidget(self.camera_label)

        self.prediction_label = QLabel("Prediction: -")
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.prediction_label)

        test_row = QHBoxLayout()
        self.test_mode_checkbox = QCheckBox("Test mode (disable action calls)")
        self.test_mode_checkbox.setChecked(True)
        self.test_mode_checkbox.toggled.connect(self._on_test_mode_toggled)
        test_row.addStretch()
        test_row.addWidget(self.test_mode_checkbox)
        test_row.addStretch()
        right_layout.addLayout(test_row)

        # Training: record the current gesture live or import a video, then retrain.
        self.record_name_edit = QLineEdit()
        self.record_name_edit.setPlaceholderText("Gesture name to record/train")
        right_layout.addWidget(self.record_name_edit)

        record_row = QHBoxLayout()
        self.record_btn = QPushButton(f"Record ({MAX_RECORD_SECONDS}s max)")
        self.record_btn.clicked.connect(self._start_recording)
        record_row.addWidget(self.record_btn)

        self.stop_record_btn = QPushButton("Stop")
        self.stop_record_btn.setEnabled(False)
        self.stop_record_btn.clicked.connect(self._stop_recording)
        record_row.addWidget(self.stop_record_btn)

        self.import_btn = QPushButton("Import video…")
        self.import_btn.clicked.connect(self._import_video)
        record_row.addWidget(self.import_btn)
        right_layout.addLayout(record_row)

        self.train_status_label = QLabel("")
        self.train_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.train_status_label)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

    def start_inference(self):
        self.worker = InferenceWorker()
        self.worker.test_mode = self.test_mode_checkbox.isChecked()
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.prediction_ready.connect(self._on_prediction)
        self.worker.recording_finished.connect(self._on_recording_finished)
        self.worker.start()

    def _start_recording(self):
        name = self.record_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Gesture name required", "Type a gesture name first.")
            return
        if not self.worker:
            self.worker = InferenceWorker()
            self.worker.test_mode = self.test_mode_checkbox.isChecked()
            self.worker.frame_ready.connect(self._on_frame)
            self.worker.prediction_ready.connect(self._on_prediction)
            self.worker.recording_finished.connect(self._on_recording_finished)
            self.worker.start()
        self.worker.start_recording(name)
        self.record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(True)
        self.train_status_label.setText(f"Recording '{name}'…")

    def _stop_recording(self):
        if self.worker:
            self.worker.stop_recording()

    def _on_recording_finished(self, name, count):
        self.record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        ensure_gesture(name)
        self.gestures_panel.reload()
        self.train_status_label.setText(f"Captured {count} samples for '{name}'. Training…")
        self._retrain()

    def _import_video(self):
        name = self.record_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Gesture name required", "Type a gesture name first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select gesture video")
        if not path:
            return
        count = import_video_file(name, path)
        ensure_gesture(name)
        self.gestures_panel.reload()
        self.train_status_label.setText(f"Captured {count} samples for '{name}'. Training…")
        self._retrain()

    def _retrain(self):
        self._train_thread = TrainThread()
        self._train_thread.finished_ok.connect(self._on_train_ok)
        self._train_thread.failed.connect(self._on_train_failed)
        self._train_thread.start()

    def _on_train_ok(self):
        self.train_status_label.setText("Training complete. Reloading model…")
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
        self.start_inference()
        self.train_status_label.setText("Model updated.")

    def _on_train_failed(self, message):
        self.train_status_label.setText(f"Training failed: {message}")

    def reset_model(self):
        """Called after a fresh-start reset: stop inference since there is no model/gesture left."""
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None
        self.camera_label.setText("No model. Record a gesture to start.")
        self.prediction_label.setText("Prediction: -")
        self.train_status_label.setText("Everything reset.")

    def _on_test_mode_toggled(self, checked):
        if self.worker:
            self.worker.test_mode = checked

    def _on_frame(self, qimg):
        pix = QPixmap.fromImage(qimg).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio)
        self.camera_label.setPixmap(pix)

    def _on_prediction(self, label, confidence):
        self.prediction_label.setText(f"Prediction: {label} ({confidence:.2f})")

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
        super().closeEvent(event)
