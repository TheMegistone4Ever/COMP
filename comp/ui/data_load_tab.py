from os.path import exists

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox

from comp.io import load_center_data_from_json


class DataLoadTab(QWidget):
    data_loaded = pyqtSignal(object)
    status_updated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.info_label = None
        self.load_button = None
        self.status_label = None
        self.loaded_filepath = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.info_label = QLabel("Завантажте файл даних центру у форматі .json")
        layout.addWidget(self.info_label)

        self.load_button = QPushButton("Огляд...")
        self.load_button.clicked.connect(self.load_data_file)  # type: ignore
        layout.addWidget(self.load_button)
        layout.addStretch(1)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def load_data_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filepath, _ = QFileDialog.getOpenFileName(self, "Завантажити дані центру", "",
                                                  "JSON Files (*.json);;All Files (*)", options=options)
        if filepath:
            if not exists(filepath):
                QMessageBox.warning(self, "Помилка", f"Файл не знайдено: {filepath}")
                self.status_updated.emit(f"Помилка: Файл не знайдено {filepath}")  # type: ignore
                return

            try:
                center_data = load_center_data_from_json(filepath)
                self.loaded_filepath = filepath
                self.data_loaded.emit(center_data)  # type: ignore
                self.status_label.setText(f"Файл завантажено: {filepath.split("/")[-1]}")
                self.status_updated.emit(f"Дані успішно завантажені з файлу: {filepath}")  # type: ignore
            except Exception as e:
                QMessageBox.critical(self, "Помилка завантаження", f"Не вдалося завантажити дані: {str(e)}")
                self.status_label.setText("Помилка завантаження файлу.")
                self.status_updated.emit(f"Помилка завантаження даних: {str(e)}")  # type: ignore
                self.data_loaded.emit(None)  # type: ignore
