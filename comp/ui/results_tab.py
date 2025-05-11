from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QFileDialog, QMessageBox, \
    QApplication


class ResultsTab(QWidget):
    status_updated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_display = None
        self.copy_button = None
        self.save_button = None
        self.results_text = ""
        self.solver_instance = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        layout.addWidget(self.results_display, 1)

        buttons_layout = QHBoxLayout()
        self.copy_button = QPushButton("Скопіювати в буфер обміну")
        self.copy_button.clicked.connect(self.copy_results)  # type: ignore
        self.copy_button.setEnabled(False)
        buttons_layout.addWidget(self.copy_button)

        self.save_button = QPushButton("Зберегти у файл .json")
        self.save_button.clicked.connect(self.save_results)  # type: ignore
        self.save_button.setEnabled(False)
        buttons_layout.addWidget(self.save_button)

        layout.addLayout(buttons_layout)

    def display_results(self, results_text: str, solver_instance):
        self.results_text = results_text
        self.solver_instance = solver_instance
        self.results_display.setText(results_text)
        self.copy_button.setEnabled(bool(results_text))
        self.save_button.setEnabled(bool(solver_instance))
        if results_text:
            self.status_updated.emit("Результати розрахунку відображені.")  # type: ignore
        else:
            self.status_updated.emit("Результати розрахунку відсутні або очищені.")  # type: ignore

    def copy_results(self):
        if self.results_text:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.results_text)
            QMessageBox.information(self, "Скопійовано", "Результати скопійовано до буфера обміну.")
            self.status_updated.emit("Результати скопійовано до буфера обміну.")  # type: ignore
        else:
            QMessageBox.warning(self, "Немає даних", "Немає результатів для копіювання.")

    def save_results(self):
        if not self.solver_instance:
            QMessageBox.warning(self, "Немає даних", "Немає результатів для збереження.")
            return

        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(self, "Зберегти результати", "",
                                                  "JSON Files (*.json);;All Files (*)", options=options)
        if filepath:
            try:
                self.solver_instance.save_results_to_json(filepath)
                QMessageBox.information(self, "Збережено", f"Результати збережено до файлу: {filepath}")
                self.status_updated.emit(f"Результати збережено до файлу: {filepath}")  # type: ignore
            except Exception as e:
                QMessageBox.critical(self, "Помилка збереження", f"Не вдалося зберегти результати: {str(e)}")
                self.status_updated.emit(f"Помилка збереження результатів: {str(e)}")  # type: ignore

    def clear_results(self):
        self.results_text = ""
        self.solver_instance = None
        self.results_display.clear()
        self.copy_button.setEnabled(False)
        self.save_button.setEnabled(False)
