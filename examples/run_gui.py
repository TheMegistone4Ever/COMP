from multiprocessing import freeze_support
from os import path
from sys import path as sys_path, argv, exit as sys_exit

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
    sys_exit(app.exec_())


if __name__ == "__main__":
    """
    This script is used to build the GUI application on Windows 11:

    pyinstaller --onefile --windowed --name "COMP" --clean `
    --upx-dir="C:\\upx-4.2.4" `
    --upx-exclude _uuid.pyd `
    --upx-exclude python3.dll `
    --version-file version.txt `
    --add-data "./comp/media/COMP.ico:comp/media" `
    --icon="./comp/media/COMP.ico" `
    --collect-all ortools `
    --distpath dist_ui `
    --workpath build_ui `
    examples/run_gui.py
    """

    freeze_support()
    main_app()
