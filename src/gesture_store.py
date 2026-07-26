"""Load/save gesture definitions (action, keybinding, mode, cooldown) to JSON."""
import json
from pathlib import Path
from config import settings

GESTURES_PATH = Path(settings.get('gesture_settings', {}).get('store_path', './config/gestures.json'))

DEFAULT_GESTURE = {
    "name": "",
    "action_type": "script",   # "script" | "keypress"
    "action_value": "",        # shell command/path, or "ctrl+space"
    "key_mode": "normal",      # "normal" | "pulse"
    "presses_per_second": 2,   # pulse mode only
    "cooldown": settings.get('gesture_settings', {}).get('cooldown_seconds', 2),
}


def load_gestures() -> list:
    if not GESTURES_PATH.exists():
        return []
    with open(GESTURES_PATH, 'r') as f:
        return json.load(f)


def save_gestures(gestures: list):
    GESTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GESTURES_PATH, 'w') as f:
        json.dump(gestures, f, indent=2)


def reset_all():
    """Deletes all gesture bindings."""
    save_gestures([])


def ensure_gesture(name: str):
    """Adds a default binding entry for `name` if it doesn't exist yet (after recording/training)."""
    gestures = load_gestures()
    if any(g["name"] == name for g in gestures):
        return gestures
    gesture = dict(DEFAULT_GESTURE)
    gesture["name"] = name
    gestures.append(gesture)
    save_gestures(gestures)
    return gestures
