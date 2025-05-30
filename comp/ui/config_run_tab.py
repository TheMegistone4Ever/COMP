from dataclasses import replace

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSpinBox, QComboBox, QPushButton,
                             QTextEdit, QProgressBar, QMessageBox, QListWidget, QListWidgetItem, QCheckBox)

from comp.models import CenterType, CenterData
from comp.utils import stringify


class ConfigRunTab(QWidget):
    """
    QWidget tab for configuring and running calculations.

    Displays center settings, lists elements for detail view, and has a
    "Run Calculation" button.
    Shows progress via a QProgressBar.
    Emits `run_calculation_requested` and `status_updated` signals.
    """

    run_calculation_requested = pyqtSignal(object)
    status_updated = pyqtSignal(str)

    def __init__(self, parent=None):
        """Initializes the ConfigRunTab."""

        super().__init__(parent)

        self.threads_spinbox = None
        self.threshold_spinbox = None
        self.type_combobox = None
        self.select_all_checkbox = None
        self.elements_list_widget = None
        self.data_display_textedit = None
        self.run_button = None
        self.progress_bar = None
        self.center_data = None
        self.init_ui()

    def init_ui(self):
        """
        Initializes UI: element selection, config group, data display,
        run button, progress bar.
        """

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        top_row_layout = QHBoxLayout()
        top_row_layout.setSpacing(15)

        elements_selection_group = QGroupBox("Вибір елементів для відображення")
        elements_selection_layout = QVBoxLayout()

        self.select_all_checkbox = QCheckBox("Обрати всі елементи для відображення")
        self.select_all_checkbox.stateChanged.connect(self.toggle_all_elements_selection)  # type: ignore
        elements_selection_layout.addWidget(self.select_all_checkbox)

        self.elements_list_widget = QListWidget()
        self.elements_list_widget.itemChanged.connect(self.update_data_display_from_selection)  # type: ignore
        elements_selection_layout.addWidget(self.elements_list_widget, 1)

        elements_selection_group.setLayout(elements_selection_layout)
        top_row_layout.addWidget(elements_selection_group, 1)

        config_group = QGroupBox("Налаштування Центру")
        config_layout = QVBoxLayout()

        threads_layout = QHBoxLayout()
        threads_label = QLabel("Кількість потоків:")
        self.threads_spinbox = QSpinBox()
        self.threads_spinbox.setMinimum(1)
        self.threads_spinbox.setMaximum(128)
        threads_layout.addWidget(threads_label)
        threads_layout.addWidget(self.threads_spinbox)
        threads_layout.addStretch()
        config_layout.addLayout(threads_layout)

        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Мін. поріг паралелізації:")
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setMinimum(1)
        self.threshold_spinbox.setMaximum(1000)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_spinbox)
        threshold_layout.addStretch()
        config_layout.addLayout(threshold_layout)

        type_layout = QHBoxLayout()
        type_label = QLabel("Тип центру:")
        self.type_combobox = QComboBox()
        for center_type in CenterType:
            self.type_combobox.addItem(center_type.name, center_type)
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_combobox)
        type_layout.addStretch()
        config_layout.addLayout(type_layout)

        config_layout.addStretch(1)

        config_group.setLayout(config_layout)
        top_row_layout.addWidget(config_group, 1)

        main_layout.addLayout(top_row_layout, 1)

        data_details_group = QGroupBox("Дворівнева організаційно-виробнича система")
        data_details_layout = QVBoxLayout()

        self.data_display_textedit = QTextEdit()
        self.data_display_textedit.setReadOnly(True)
        data_details_layout.addWidget(self.data_display_textedit, 1)

        data_details_group.setLayout(data_details_layout)
        main_layout.addWidget(data_details_group, 2)

        self.run_button = QPushButton("Запустити розрахунок")
        self.run_button.clicked.connect(self.request_calculation)  # type: ignore
        self.run_button.setEnabled(False)
        main_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

    def update_config_display(self, center_data: CenterData):
        """
        Populates UI with data from `center_data` or clears it.

        :param center_data: `CenterData` to display, or None.
        """

        self.center_data = center_data
        if center_data:
            self.threads_spinbox.setValue(center_data.config.num_threads or 1)
            self.threshold_spinbox.setValue(center_data.config.min_parallelisation_threshold or 1)

            index = self.type_combobox.findData(center_data.config.type)
            if index >= 0:
                self.type_combobox.setCurrentIndex(index)

            self.elements_list_widget.clear()
            for i, element in enumerate(center_data.elements):
                item = QListWidgetItem(f"Елемент {element.config.id} ({element.config.type.name})")
                item.setData(Qt.UserRole, element)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.elements_list_widget.addItem(item)

            self.select_all_checkbox.setCheckState(Qt.Unchecked)
            self.update_data_display()
            self.run_button.setEnabled(True)
            self.status_updated.emit("Налаштування та дані центру готові до розрахунку.")  # type: ignore
        else:
            self.data_display_textedit.clear()
            self.elements_list_widget.clear()
            if self.threads_spinbox:
                self.threads_spinbox.setValue(1)
                self.threshold_spinbox.setValue(1)
                self.type_combobox.setCurrentIndex(0) if self.type_combobox.count() > 0 else None
                self.select_all_checkbox.setCheckState(Qt.Unchecked)
            self.run_button.setEnabled(False)
            self.status_updated.emit("Завантажте дані для налаштування та розрахунку.")  # type: ignore

    def toggle_all_elements_selection(self, state):
        """Handles "Select All" checkbox state change for an element list."""

        is_checked = (state == Qt.Checked)
        for i in range(self.elements_list_widget.count()):
            item = self.elements_list_widget.item(i)
            item.setCheckState(Qt.Checked if is_checked else Qt.Unchecked)

    def update_data_display_from_selection(self, item_changed):
        """Updates data display when an element’s check state changes."""

        print(f"Item changed: {item_changed.text()}")
        self.update_data_display()

    def update_data_display(self):
        """
        Updates the data display QTextEdit based on selected elements
        or shows overall center data if none are selected.
        Manages "Select All" checkbox state.
        """

        if not self.center_data:
            self.data_display_textedit.clear()
            return

        selected_elements_data = list()
        any_selected = False
        all_selected_or_none_in_list = True

        if self.elements_list_widget.count() == 0:
            all_selected_or_none_in_list = True
        else:
            for i in range(self.elements_list_widget.count()):
                item = self.elements_list_widget.item(i)
                if item.checkState() == Qt.Checked:
                    selected_elements_data.append(item.data(Qt.UserRole))
                    any_selected = True
                else:
                    all_selected_or_none_in_list = False

        # Update "Select All" checkbox state without triggering its own signal
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setCheckState(
            Qt.Checked if all_selected_or_none_in_list and self.elements_list_widget.count() > 0 else Qt.Unchecked)
        self.select_all_checkbox.blockSignals(False)

        if any_selected:
            display_text = "Обрані елементи:\n"
            for el_data in selected_elements_data:
                display_text += f"Елемент ID: {el_data.config.id}:\n"
                display_text += stringify(el_data, precision=4) + "\n\n"
            self.data_display_textedit.setText(display_text)
        else:
            self.data_display_textedit.setText(stringify(self.center_data, precision=4))

    def request_calculation(self):
        """
        Gathers UI configurations and emits `run_calculation_requested`.
        Updates UI for running state (button, progress bar).
        """

        if not self.center_data:
            QMessageBox.warning(self, "Дані не завантажені", "Будь ласка, завантажте дані центру перед запуском.")
            return

        updated_config = replace(
            self.center_data.config,
            num_threads=self.threads_spinbox.value(),
            min_parallelisation_threshold=self.threshold_spinbox.value(),
            type=self.type_combobox.currentData()
        )
        modified_center_data = replace(self.center_data, config=updated_config)

        self.run_calculation_requested.emit(modified_center_data)  # type: ignore
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_updated.emit("Запуск розрахунку...")  # type: ignore

    def calculation_finished(self, success: bool):
        """
        Updates UI after calculation (button, progress bar, status).

        :param success: True if the calculation was successful.
        """

        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        if success:
            self.status_updated.emit("Розрахунок завершено.")  # type: ignore
        else:
            self.status_updated.emit("Розрахунок завершено з помилкою.")  # type: ignore

    def set_progress(self, value):
        """Sets the progress bar value."""

        self.progress_bar.setValue(value)
