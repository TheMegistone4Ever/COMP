from PyQt5.QtCore import QThread, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QMessageBox, QStatusBar

from comp.models import CenterData
from comp.ui.config_run_tab import ConfigRunTab
from comp.ui.data_load_tab import DataLoadTab
from comp.ui.results_tab import ResultsTab
from comp.ui.styles import STYLESHEET
from comp.ui.worker import SolverWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_load_tab = None
        self.config_run_tab = None
        self.results_tab = None
        self.status_bar = None
        self.tab_widget = None
        self.center_data = None
        self.solver_instance = None
        self.results_text_data = None
        self.results_dict_data = None
        self.solver_thread = None
        self.solver_worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("УЗГОДЖЕНЕ ПЛАНУВАННЯ В ДВОРІВНЕВИХ ОРГАНІЗАЦІЙНО-ВИРОБНИЧИХ СИСТЕМАХ")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet(STYLESHEET)

        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        self.data_load_tab = DataLoadTab()
        self.config_run_tab = ConfigRunTab()
        self.results_tab = ResultsTab()

        self.tab_widget.addTab(self.data_load_tab, "1. Завантаження даних")
        self.tab_widget.addTab(self.config_run_tab, "2. Налаштування та Розрахунок")
        self.tab_widget.addTab(self.results_tab, "3. Перегляд результатів")

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готово до роботи.")

        self.data_load_tab.data_loaded.connect(self.on_data_loaded)
        self.config_run_tab.run_calculation_requested.connect(self.run_calculation)

        self.data_load_tab.status_updated.connect(self.status_bar.showMessage)
        self.config_run_tab.status_updated.connect(self.status_bar.showMessage)
        self.results_tab.status_updated.connect(self.status_bar.showMessage)

    @pyqtSlot(object)
    def on_data_loaded(self, center_data: CenterData):
        self.center_data = center_data
        self.solver_instance = None
        self.results_text_data = None
        self.results_dict_data = None

        self.config_run_tab.update_config_display(center_data)
        self.results_tab.clear_results()
        if center_data:
            self.tab_widget.setCurrentWidget(self.config_run_tab)

    @pyqtSlot(object)
    def run_calculation(self, modified_center_data: CenterData):
        if self.solver_thread and self.solver_thread.isRunning():
            QMessageBox.information(self, "Розрахунок триває", "Будь ласка, зачекайте завершення поточного розрахунку.")
            return

        self.center_data = modified_center_data
        self.solver_thread = QThread()
        self.solver_worker = SolverWorker(self.center_data)
        self.solver_worker.moveToThread(self.solver_thread)

        self.solver_worker.finished.connect(self.on_calculation_finished)
        self.solver_worker.error.connect(self.on_calculation_error)
        self.solver_worker.progress.connect(self.config_run_tab.set_progress)

        self.solver_thread.started.connect(self.solver_worker.run)  # type: ignore
        self.solver_thread.finished.connect(self.solver_thread.deleteLater)  # type: ignore
        self.solver_worker.finished.connect(self.solver_thread.quit)
        self.solver_worker.error.connect(self.solver_thread.quit)

        self.solver_thread.start()

    @pyqtSlot(object, str, dict, str)
    def on_calculation_finished(self, solver_instance, results_text, results_dict, status_message):
        self.solver_instance = solver_instance
        self.results_text_data = results_text
        self.results_dict_data = results_dict

        self.config_run_tab.calculation_finished(True)
        self.results_tab.display_results(results_text, solver_instance)
        self.status_bar.showMessage(status_message)
        self.tab_widget.setCurrentWidget(self.results_tab)

        if self.solver_thread:
            self.solver_thread.quit()
            self.solver_thread.wait()
            self.solver_thread = None
        self.solver_worker = None

    @pyqtSlot(str)
    def on_calculation_error(self, error_message):
        QMessageBox.critical(self, "Помилка розрахунку", error_message)
        self.config_run_tab.calculation_finished(False)
        self.status_bar.showMessage(f"Помилка: {error_message}")
        if self.solver_thread:
            self.solver_thread.quit()
            self.solver_thread.wait()
            self.solver_thread = None
        self.solver_worker = None

    def closeEvent(self, event):
        if self.solver_thread and self.solver_thread.isRunning():
            reply = QMessageBox.question(self, "Вихід", "Розрахунок ще триває. Ви впевнені, що хочете вийти?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.solver_thread.quit()
                self.solver_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
