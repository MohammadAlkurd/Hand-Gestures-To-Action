# Hand Gestures AI

A PyQt6 desktop app that recognizes custom hand gestures from your webcam and binds them to keypresses or scripts. Record a gesture, train a classifier on it, assign an action, and the app fires that action live whenever it detects the gesture.

## How it works

1. **Record** — capture a short video (webcam or imported file) of yourself performing a gesture.
2. **Train** — hand landmarks are extracted (via a hand landmark model) and a small MLP classifier is trained/updated on them.
3. **Bind** — assign each gesture to a keypress or a custom action in the gestures panel.
4. **Run** — with Test mode off, the inference loop dispatches the bound action live as gestures are recognized.

## UI overview

- **Gestures panel**: list of gesture → action bindings (Add / Edit / Remove / Reset everything).
- **Camera preview**: live feed with the current prediction and confidence shown below it.
- **Test mode**: when checked, predictions are shown but no actions are triggered — safe for trying out new gestures.
- **Record / Import video**: record live (5s max) or import an existing video file as training data.

## Project structure

```
hand_gestures_ai/
├── main.py                    # app entry point, system tray + QApplication setup
├── config.py                  # settings, GestureMLP model definition, device setup
├── settings.yaml              # user/app settings
├── gesture_model.pth          # trained model weights
├── hand_landmarker.task       # hand landmark detection model
├── hand_gestures.ipynb        # exploration / experimentation notebook
├── cache/                     # cached landmark samples per gesture
├── config/                    # config module
├── models/                    # saved models
└── src/
    ├── actions.py              # action_manager: dispatches keypress/script actions
    ├── dataset_loader.py       # landmark extraction, normalization, dataset loading
    ├── gesture_store.py        # load/save/ensure gesture bindings
    ├── inference.py             # InferenceWorker: real-time landmark -> prediction -> action
    ├── keys.py                  # keypress definitions (pynput)
    ├── recorder.py               # webcam recording + video import, sample saving
    ├── trainer.py                 # MLP training loop
    └── ui/
        ├── main_window.py         # MainWidget: top-level layout
        ├── gestures_panel.py      # gesture bindings list panel
        └── gesture_dialog.py      # add/edit gesture dialog
```

## Requirements

- Python 3.14
- PyQt6
- torch
- mediapipe
- opencv-python (`cv2`)
- pynput
- PyYAML
- numpy

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

1. In the gestures panel, click **Add** and give your gesture a name.
2. Click **Record (5s max)** and perform the gesture in front of the camera (or **Import video...** to use existing footage).
3. Train the model on the recorded samples.
4. Assign a keypress or script action to the gesture.
5. Uncheck **Test mode** to enable live actions from the system tray app.

## Disclaimer

This project was built with substantial assistance from **Claude Sonnet 5** (Anthropic), used as an AI pair-programmer for design, implementation, and debugging throughout development. Review the code before using it in any sensitive or production environment.

## License

MIT .
