from multiprocessing import freeze_support
from os import path
from sys import path as sys_path, argv, exit

from PyQt5.QtWidgets import QApplication

project_root = path.abspath(path.join(path.dirname(__file__), ".."))
if project_root not in sys_path:
    sys_path.insert(0, project_root)

from comp.ui import MainWindow


def main_app():
    """Main function to run the GUI application."""

    app = QApplication(argv)
    main_win = MainWindow()
    main_win.show()
    exit(app.exec_())


if __name__ == "__main__":
    """
    This script is used to run the GUI application:

    pyinstaller --onefile --windowed --name "COMP" --clean `
    --upx-dir="C:\\upx-4.2.4" `
    --version-file version.txt `
    --add-data "comp/media/COMP.ico:comp/media" `
    --distpath dist_ui `
    --workpath build_ui `
    examples/run_gui.py
    """

    freeze_support()
    main_app()
