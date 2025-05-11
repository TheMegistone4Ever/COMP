STYLESHEET = """
QMainWindow {
    background-color: #F5F5F5;
}
QTabWidget::pane {
    border-top: 1px solid #C8C8C8;
    background-color: #F5F5F5;
}
QTabBar::tab {
    background: #E1DFDD;
    color: #323130;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: 500;
    border: 1px solid #C8C8C8;
    border-bottom: none;
    min-width: 150px;
}
QTabBar::tab:selected {
    background: #F5F5F5;
    border-bottom: 2px solid #F5F5F5; /* Effectively hides bottom border against pane */
}
QTabBar::tab:hover {
    background: #D0D0D0;
}
QPushButton {
    background-color: #0078D4;
    color: white;
    border: none;
    padding: 10px 20px;
    font-weight: 500;
    margin: 4px;
    font-size: 16px;
    min-height: 30px;
}
QPushButton:hover {
    background-color: #106EBE;
}
QPushButton:disabled {
    background-color: #C8C8C8;
    color: #8A8886;
}
QLabel {
    font-size: 18px;
    font-weight: bold;
    color: #323130;
    padding-bottom: 8px;
}
QSpinBox, QDoubleSpinBox {
    padding: 6px;
    border: 1px solid #8A8886;
    background: white;
    min-width: 100px;
    font-size: 16px;
    min-height: 28px;
}
QComboBox {
    padding: 6px;
    border: 1px solid #8A8886;
    background: white;
    font-size: 16px;
    min-width: 180px;
    min-height: 28px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox::down-arrow {
    image: url(none); 
}
QTextEdit {
    padding: 8px;
    border: 1px solid #8A8886;
    background: white;
    font-size: 14px;
    font-family: "Consolas", "Monaco", monospace;
}
QProgressBar {
    border: none;
    background-color: #E1DFDD;
    height: 10px;
    text-align: center;
    color: #323130;
    font-size: 10px;
}
QProgressBar::chunk {
    background-color: #0078D4;
}
QMessageBox {
    font-size: 16px;
}
QMessageBox QLabel {
    font-size: 16px;
    font-weight: normal;
}
QMessageBox QPushButton {
    padding: 10px 20px;
    min-width: 80px;
    font-size: 14px;
}
QCheckBox {
    font-size: 16px;
    spacing: 8px;
}
QListWidget {
    font-size: 14px;
    border: 1px solid #8A8886;
    background: white;
}
QGroupBox {
    font-size: 16px;
    font-weight: bold;
    color: #323130;
    border: 1px solid #C8C8C8;
    margin-top: 10px;
    padding-top: 20px;
    padding-left: 10px;
    padding-right: 10px;
    padding-bottom: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    left: 10px;
}
QLineEdit {
    padding: 6px;
    border: 1px solid #8A8886;
    background: white;
    font-size: 16px;
    min-height: 28px;
}
"""
