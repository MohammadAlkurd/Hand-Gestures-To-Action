"""Keyboard key names and hotkey helpers, built on pynput."""
from pynput.keyboard import Key, KeyCode, Controller

# Special keys the user can reference by name (e.g. "ctrl+space").
SPECIAL_KEYS = {
    "ctrl": Key.ctrl, "alt": Key.alt, "shift": Key.shift, "cmd": Key.cmd,
    "space": Key.space, "enter": Key.enter, "tab": Key.tab, "esc": Key.esc,
    "backspace": Key.backspace, "delete": Key.delete, "up": Key.up,
    "down": Key.down, "left": Key.left, "right": Key.right,
    "home": Key.home, "end": Key.end, "page_up": Key.page_up,
    "page_down": Key.page_down, "caps_lock": Key.caps_lock,
    "f1": Key.f1, "f2": Key.f2, "f3": Key.f3, "f4": Key.f4, "f5": Key.f5,
    "f6": Key.f6, "f7": Key.f7, "f8": Key.f8, "f9": Key.f9, "f10": Key.f10,
    "f11": Key.f11, "f12": Key.f12,
    "vol_up" : Key.media_volume_up, "vol_down": Key.media_volume_down,
    "mute": Key.media_volume_mute,
    "stop_video":Key.media_play_pause,
    "next_video":Key.media_next,
    "prev_video":Key.media_previous,
}

# Pseudo-actions that are not real key presses but built-in app behaviors.
BUILTIN_ACTIONS = ["stop_video", "toggle_test_mode"]

# Full reference list shown to the user (name -> id to type in the UI).
AVAILABLE_KEY_NAMES = sorted(SPECIAL_KEYS.keys()) + list("abcdefghijklmnopqrstuvwxyz0123456789")

_controller = Controller()


def _resolve(token: str):
    token = token.strip().lower()
    if token in SPECIAL_KEYS:
        return SPECIAL_KEYS[token]
    if len(token) == 1:
        return KeyCode.from_char(token)
    raise ValueError(f"Unknown key: {token}")


def parse_hotkey(combo: str):
    """Turns 'ctrl+space' into a list of pynput key objects."""
    return [_resolve(part) for part in combo.split("+") if part.strip()]


def press_hotkey(combo: str):
    """Presses and releases a full key combination once."""
    keys = parse_hotkey(combo)
    for k in keys:
        _controller.press(k)
    for k in reversed(keys):
        _controller.release(k)
