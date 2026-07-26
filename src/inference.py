import cv2
import torch
import time
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from src.dataset_loader import extract_landmarks, normalize_landmarks
from config import settings, GestureMLP, device
from src.gesture_store import load_gestures
from src.actions import action_manager
from src.recorder import save_landmark_sample

NEUTRAL_LABELS = {"neutral", "none", "idle"}
RECORD_MAX_SECONDS = 5
RECORD_FRAME_SKIP = 2


class InferenceWorker(QThread):
    """Runs the camera + model loop in a background thread for the UI."""

    frame_ready = pyqtSignal(QImage)
    prediction_ready = pyqtSignal(str, float)

    recording_finished = pyqtSignal(str, int)

    def __init__(self, video_path=None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self._running = False
        self.test_mode = True
        self._current_gesture = None
        self._current_gesture_start = 0
        self.recording = False
        self._record_name = None
        self._record_start = 0
        self._record_frame_idx = 0
        self._record_count = 0

    def stop(self):
        self._running = False

    def start_recording(self, gesture_name: str):
        """Starts caching landmarks for `gesture_name` for up to RECORD_MAX_SECONDS."""
        self._record_name = gesture_name
        self._record_start = time.time()
        self._record_frame_idx = 0
        self._record_count = 0
        self.recording = True

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.recording_finished.emit(self._record_name, self._record_count)

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(0 if self.video_path is None else str(self.video_path))
        infer_model, inv_label_map = None, {}
        checkpoint_path = Path(settings['model']['checkpoint_path'])
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            inv_label_map = {v: k for k, v in checkpoint['label_map'].items()}
            infer_model = GestureMLP(num_classes=len(inv_label_map)).to(device)
            infer_model.load_state_dict(checkpoint['model_state_dict'])
            infer_model.eval()

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = extract_landmarks(frame_rgb)

            label_name, confidence = "no hand", 0.0
            if landmarks is not None and infer_model is not None:
                features = normalize_landmarks(landmarks)
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = infer_model(features_tensor)
                    pred = output.argmax(1).item()
                    confidence = torch.softmax(output, dim=1)[0, pred].item()
                label_name = inv_label_map[pred]

            if self.recording:
                self._record_frame_idx += 1
                if landmarks is not None and self._record_frame_idx % RECORD_FRAME_SKIP == 0:
                    save_landmark_sample(self._record_name, landmarks)
                    self._record_count += 1
                if time.time() - self._record_start >= RECORD_MAX_SECONDS:
                    self.stop_recording()

            self._handle_gesture(label_name)
            self.prediction_ready.emit(label_name, confidence)

            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.frame_ready.emit(qimg.copy())

            time.sleep(0.01)

        cap.release()

    def _handle_gesture(self, label_name: str):
        if label_name == self._current_gesture:
            gesture = self._find_gesture(label_name)
            if not self.test_mode and gesture:
                held_for = time.time() - self._current_gesture_start
                if held_for >= gesture.get("start_after", 0):
                    action_manager.notify_detected(gesture)
            return

        # Gesture changed: stop pulsing the previous one.
        if self._current_gesture:
            action_manager.notify_not_detected(self._current_gesture)
        self._current_gesture = label_name
        self._current_gesture_start = time.time()

    @staticmethod
    def _find_gesture(name: str):
        if name.lower() in NEUTRAL_LABELS:
            return None
        for g in load_gestures():
            if g["name"] == name:
                return g
        return None


def run_inference(video_path=None):
    """Legacy blocking CLI runner (prints triggers instead of executing actions)."""
    cap = cv2.VideoCapture(0 if video_path is None else str(video_path))
    checkpoint = torch.load(settings['model']['checkpoint_path'], map_location=device)
    inv_label_map = {v: k for k, v in checkpoint['label_map'].items()}
    model = GestureMLP(num_classes=len(inv_label_map)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = extract_landmarks(frame_rgb)

        if landmarks is not None:
            features = normalize_landmarks(landmarks)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(features_tensor)
                pred = output.argmax(1).item()
                confidence = torch.softmax(output, dim=1)[0, pred].item()

            label_text = f"{inv_label_map[pred]} ({confidence:.2f})"
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Gesture Recognizer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
