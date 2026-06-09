
import sys
from PyQt6.QtWidgets import QApplication

# Run every script
if __name__ == "__main__":
    from scripts.default_gui import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

