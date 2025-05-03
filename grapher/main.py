import sys
import igraph as ig
from PyQt6.QtWidgets import QApplication
from gui import MainWindow

if __name__ == "__main__":
    ig.config['plotting.backend'] = 'matplotlib'
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())