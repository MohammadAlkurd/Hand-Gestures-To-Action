import sys

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QAction

from src.ui.main_window import MainWidget
from config import settings


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    window = MainWidget()
    window.setWindowTitle(settings.get('ui', {}).get('window_title', 'Gesture Recognizer UI'))
    window.resize(*settings.get('ui', {}).get('min_window_size', [800, 600]))
    window.show()

    # System tray icon: closing the window only hides it, the app keeps running.
    tray = QSystemTrayIcon(QIcon.fromTheme("input-gaming"), app)
    tray.setToolTip(settings.get('ui', {}).get('window_title', 'Gesture Recognizer UI'))

    menu = QMenu()
    show_action = QAction("Show")
    show_action.triggered.connect(window.showNormal)
    show_action.triggered.connect(window.activateWindow)
    menu.addAction(show_action)

    hide_action = QAction("Hide")
    hide_action.triggered.connect(window.hide)
    menu.addAction(hide_action)

    toggle_action = QAction("Test Mode", checkable=True)
    toggle_action.setChecked(window.test_mode_checkbox.isChecked())
    toggle_action.toggled.connect(window.test_mode_checkbox.setChecked)
    window.test_mode_checkbox.toggled.connect(toggle_action.setChecked)
    menu.addAction(toggle_action)

    menu.addSeparator()
    quit_action = QAction("Quit")

    def quit_app():
        if window.worker:
            window.worker.stop()
            window.worker.wait(1000)
        app.quit()

    quit_action.triggered.connect(quit_app)
    menu.addAction(quit_action)

    tray.setContextMenu(menu)
    tray.activated.connect(
        lambda reason: window.showNormal() if reason == QSystemTrayIcon.ActivationReason.Trigger else None
    )
    tray.show()

    # Closing the window (X button) hides it instead of quitting the app.
    def close_to_tray(event):
        event.ignore()
        window.hide()

    window.closeEvent = close_to_tray

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
