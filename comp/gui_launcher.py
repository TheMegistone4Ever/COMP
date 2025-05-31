from multiprocessing import freeze_support
from sys import argv, exit as sys_exit

from PyQt5.QtWidgets import QApplication

from comp.ui import MainWindow


def main_app_entry():
    freeze_support()
    app = QApplication(argv)
    main_win = MainWindow()
    main_win.show()
    sys_exit(app.exec_())


if __name__ == "__main__":
    main_app_entry()
