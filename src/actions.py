"""Executes gesture actions: run a script/app, or press a key (normal/pulse)."""
import subprocess
import threading
import time

from src.keys import press_hotkey


class ActionManager:
    """Tracks per-gesture cooldowns and pulse threads, dispatches actions."""

    def __init__(self):
        self._last_trigger = {}
        self._pulse_stop_events = {}

    def notify_detected(self, gesture: dict):
        """Call every frame the gesture is the current prediction (test_mode=False)."""
        name = gesture["name"]
        now = time.time()
        cooldown = gesture.get("cooldown", 2)

        if gesture.get("action_type") == "keypress" and gesture.get("key_mode") == "pulse":
            if now - self._last_trigger.get(name, 0) < cooldown:
                return  # still cooling down since the gesture last stopped
            self._ensure_pulse(gesture)
            return

        if now - self._last_trigger.get(name, 0) < cooldown:
            return
        self._last_trigger[name] = now
        self._fire_once(gesture)

    def notify_not_detected(self, gesture_name: str):
        """Stops a pulse thread once the gesture is no longer being predicted, starting its cooldown."""
        ev = self._pulse_stop_events.pop(gesture_name, None)
        if ev:
            ev.set()
            self._last_trigger[gesture_name] = time.time()

    def _ensure_pulse(self, gesture: dict):
        name = gesture["name"]
        if name in self._pulse_stop_events:
            return  # already pulsing
        stop_event = threading.Event()
        self._pulse_stop_events[name] = stop_event
        interval = 1.0 / max(gesture.get("presses_per_second", 2), 0.1)

        def _loop():
            while not stop_event.is_set():
                try:
                    press_hotkey(gesture["action_value"])
                except Exception as e:
                    print(f"[actions] pulse error for {name}: {e}")
                stop_event.wait(interval)

        threading.Thread(target=_loop, daemon=True).start()

    def _fire_once(self, gesture: dict):
        action_type = gesture.get("action_type")
        value = gesture.get("action_value", "")
        try:
            if action_type == "script":
                subprocess.Popen(value, shell=True)
            elif action_type == "keypress":
                press_hotkey(value)
        except Exception as e:
            print(f"[actions] failed to run action for {gesture.get('name')}: {e}")


action_manager = ActionManager()
