from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import settings

# MediaPipe hand landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)


import random

def extract_landmarks(frame_rgb, mirror_prob=0.5):

    if random.random() < mirror_prob:
        frame_rgb = np.ascontiguousarray(frame_rgb[:, ::-1, :])

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in hand])


def normalize_landmarks(coords):
    """Translation + scale invariant normalization. Returns flat (63,) vector."""
    wrist = coords[0].copy()
    coords = coords - wrist
    scale = np.linalg.norm(coords, axis=1).max()
    coords = coords / (scale + 1e-6)
    return coords.flatten()


class VideoDataset:
    """Iterator over labeled frames from videos in a dataset folder."""

    def __init__(self, root_dir, frame_skip=5):
        self.root_dir = Path(root_dir)
        self.frame_skip = frame_skip
        self.classes = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.label_map = {name: idx for idx, name in enumerate(self.classes)}

    def __iter__(self):
        for class_name, label_idx in self.label_map.items():
            class_folder = self.root_dir / class_name
            for video_name in sorted(class_folder.iterdir()):
                if not video_name.suffix.lower() in ('.mp4', '.avi', '.mov'):
                    continue
                cap = cv2.VideoCapture(str(video_name))
                frame_count, missed = 0, 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % self.frame_skip == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        landmarks = extract_landmarks(frame_rgb)
                        if landmarks is not None:
                            yield (landmarks, label_idx)
                        else:
                            missed += 1
                    frame_count += 1
                cap.release()
                if missed and self.cache_root / class_name / video_name.name.exists():
                    print(f"  {video_name.name}: no hand detected in {missed} sampled frames")


def load_landmark_cache(cache_root=None):
    """Builds (X, y, label_map) tensors from cached per-gesture landmark .npy files."""
    import torch
    cache_root = Path(cache_root) if cache_root else Path(
        settings.get('data', {}).get('cache_root', './cache')) / 'landmarks'
    classes = sorted(d.name for d in cache_root.iterdir() if d.is_dir()) if cache_root.exists() else []
    label_map = {name: idx for idx, name in enumerate(classes)}

    features, labels = [], []
    for name, idx in label_map.items():
        for f in (cache_root / name).glob("*.npy"):
            features.append(np.load(f))
            labels.append(idx)

    X = torch.tensor(np.array(features), dtype=torch.float32) if features else torch.empty((0, 63))
    y = torch.tensor(labels, dtype=torch.long) if labels else torch.empty((0,), dtype=torch.long)
    return X, y, label_map


def load_dataset(root_dir, frame_skip=5):
    """Load all videos from a dataset root and cache outputs."""
    cache_root = Path(settings.get('data', {}).get('cache_root', './cache'))
    cache_root.mkdir(parents=True, exist_ok=True)

    loader = VideoDataset(root_dir, frame_skip)

    for features, label in loader:
        # Cache: store normalized landmarks as a NumPy array
        cache_path = cache_root / settings['data']['class_folder'] / str(label)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / f"{features.shape[0]:05d}.npy"
        np.save(cache_file, features)

    return loader