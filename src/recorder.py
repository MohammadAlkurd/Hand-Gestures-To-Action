"""Records gesture training samples as cached MediaPipe landmarks (no raw video is stored)."""
import uuid
from pathlib import Path

import cv2
import numpy as np

from config import settings
from src.dataset_loader import extract_landmarks, normalize_landmarks

CACHE_ROOT = Path(settings.get('data', {}).get('cache_root', './cache')) / 'landmarks'
MAX_RECORD_SECONDS = 5
DEFAULT_FRAME_SKIP = 2


def delete_all_cache():
    """Removes the entire landmark cache (used by the reset/fresh-start button)."""
    import shutil
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def delete_gesture_cache(gesture_name: str):
    """Removes all cached landmark samples for a gesture (used when removing it)."""
    import shutil
    folder = CACHE_ROOT / gesture_name
    if folder.exists():
        shutil.rmtree(folder)


def save_landmark_sample(gesture_name: str, landmarks) -> Path:
    """Normalizes one raw (21,3) landmark array and caches it for `gesture_name`."""
    folder = CACHE_ROOT / gesture_name
    folder.mkdir(parents=True, exist_ok=True)
    features = normalize_landmarks(landmarks)
    path = folder / f"{uuid.uuid4().hex}.npy"
    np.save(path, features)
    return path


def import_video_file(gesture_name: str, video_path,
                       frame_skip: int = DEFAULT_FRAME_SKIP,
                       max_seconds: float = MAX_RECORD_SECONDS) -> int:
    """Extracts landmarks from a user-provided video (capped at max_seconds) and caches them."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(fps * max_seconds)
    frame_idx, saved = 0, 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = extract_landmarks(frame_rgb)
            if landmarks is not None:
                save_landmark_sample(gesture_name, landmarks)
                saved += 1
        frame_idx += 1

    cap.release()
    return saved
