
from __future__ import annotations

# Standard library imports
import os
from typing import TYPE_CHECKING, Callable, Optional

# External library imports
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPixmap, QPen
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QLayout, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget)

# Custom imports
from scripts.config import IMG_DIR

# Type hinting imports
if TYPE_CHECKING:
    from scripts.default_gui import MainWindow

############################################################################

# Class for custom button attributes and behaviour
class CustomButton(QPushButton):
    def __init__(
            self, name: str, group: str, btn_type: str, parent: MainWindow,
            img: str = None, desc: str = None,
            width: int = None, height: int = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Define basic attributes
        self.name = name; self.group = group; self.parent = parent
        self.btn_type = btn_type; self.img = img
        self.parent.btns[group].append(self)

        # Logic map for name of button to function call on click
        self.actions: dict[str, Callable] = {
            # MainWindow
            "predict_btn": lambda: self.parent.predict(),
            "remove_pd_btn": lambda: self.parent.rebuild_graph(),
            # Graph
            "add_stock_btn": lambda: self.parent.add_to_graph(),
            "remove_stock_btn": lambda: self.parent.remove_from_graph(),
        }

        # Setup parameters and basic styling
        if height and width: self.setFixedSize(width, height)
        elif height: self.setFixedHeight(height)
        elif width: self.setFixedWidth(width)

        if desc: self.setToolTip(desc)
        if self.btn_type == "grp": self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.init_appearance()

        self.clicked.connect(self.handle_click)

    # Define what happens when a button is clicked
    def handle_click(self) -> None:
        # Handle group visuals
        if self.btn_type == "grp":
            for btn in self.parent.btns[self.group]:
                # Set clicked button to correct style and disable it, while enabling rest
                is_me = (btn == self)
                btn.setChecked(is_me)
                btn.setEnabled(not is_me)
                btn.update_appearance(is_active=is_me)

        # Call correct function
        action: Optional[Callable] = self.actions.get(self.name)
        if action: action()

    # External call function to reset a button to its initial state
    def reset(self):
        self.setChecked(False)
        self.setEnabled(True)
        self.update_appearance(is_active=False)

    # Initialize basic button styling
    def init_appearance(self) -> None:
        # Find correct css code depending on what type of button it is
        img_css = f"""background-image: url('{self.img}'); background-repeat: no-repeat;
        			  background-position: center;""" if self.img else ""
        press_css = "background-color: #858585;" if self.btn_type == "indv" else ""

        # Add css to button style
        self.setStyleSheet(f"""
                QPushButton {{ {img_css} background-color: #e3e3e3; border: none; font-size: 13px; 
                			   font-family: Aller display}}
                QPushButton:hover {{background-color: #adadad}}
                QPushButton:pressed {{ {press_css} }} 

                QPushButton[BorderBlank="true"] {{ {img_css} background-color: #ffffff; border: 1px solid black}}
                QPushButton[BorderBlank="true"]:hover {{background-color: #adadad}}
                QPushButton[BorderBlank="true"]:pressed {{background-color: #858585}}
    """)

    # Helper function to update button's appearance when checked
    def update_appearance(self, is_active: bool) -> None:
        background_css = f"background-color: {'#8a8a8a' if is_active else '#e3e3e3'}"
        img_css = f"""background-image: url('{self.img}'); background-repeat: no-repeat;
        			  background-position: center;""" if self.img else ""

        self.setStyleSheet(f"""
            QPushButton {{ {img_css} {background_css}; border: none; font-size: 13px; font-family: Aller display}}
            QPushButton:hover {{ {"" if is_active else "background-color: #adadad"} }} """)

############################################################################

# Helper function to create a risk slider with all styling and return layout
def create_slider_layout(parent: MainWindow) -> QVBoxLayout:
    risk_layout = QVBoxLayout(); risk_layout.setSpacing(0)

    # Create slider and label for value selected
    current_tolerance = parent.get_profile_data().get("data", {}).get("Risk tolerance", 4)
    risk_slider: QSlider = QSlider(Qt.Orientation.Horizontal)
    risk_slider.setStyleSheet("""QSlider {border: none}""")
    risk_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    risk_slider.setMinimum(1); risk_slider.setMaximum(10)
    risk_slider.setTickInterval(1); risk_slider.setSingleStep(1)
    risk_slider.setValue(current_tolerance)
    risk_slider.valueChanged.connect(
        lambda v: risk_value_label.setText(
            f"Risk tolerance: {v} {'(Current)' if v == current_tolerance else ('(Recommended)' if v == 4 else '')}"
        )
    )

    # Create labels for slider axis
    risk_value_label = QLabel(f"Risk tolerance: {current_tolerance} (Current)")
    risk_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    risk_value_label.setStyleSheet("border: none; font-size: 13px; font-family: Aller Display")
    number_layout = QHBoxLayout()
    for i in range(1, 11):
        nlabel = QLabel(str(i)); nlabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nlabel.setStyleSheet("border: none"); number_layout.addWidget(nlabel)

    # Add all layouts and widgets to main layout
    add_to_layout(risk_layout, [risk_value_label, risk_slider, number_layout])
    # Save risk slider to parent for use later and return entire layout
    parent.risk_slider = risk_slider; return risk_layout

############################################################################

# Helper class to make the circular label clickable
class ClickableLabel(QLabel):
    clicked: pyqtSignal = pyqtSignal()
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton: self.clicked.emit()
        super().mouseReleaseEvent(event)

# Helper function to create a circular label for profile icon
def create_circle_label(
        parent: MainWindow, clickable: bool = True,
        diameter: int = 100, desc: str = None, border: bool = True
) -> QLabel:
    # Get icon file and create a pixmap from it
    default = os.path.join(IMG_DIR, "person_icon.jpg").replace("\\", "/")
    pixmap = QPixmap(default)
    if pixmap.isNull(): pixmap = QPixmap(default)

    # Scale pixmap and setup painter
    pixmap = pixmap.scaled(
        diameter, diameter, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation
    )
    mask = QPixmap(diameter, diameter); mask.fill(Qt.GlobalColor.transparent)

    painter = QPainter(mask)
    path = QPainterPath(); path.addEllipse(0, 0, diameter, diameter)
    painter.setClipPath(path)

    # Draw label and border if needed
    painter.drawPixmap(0, 0, pixmap)
    if border:
        painter.setPen(QPen(QColor("#00aa00"), 3))
        painter.drawEllipse(1, 1, diameter - 3, diameter - 3)
    painter.end()

    # Make it clickable if needed
    if clickable:
        label = ClickableLabel(); label.clicked.connect(parent.label_click)
        label.setCursor(Qt.CursorShape.PointingHandCursor)
    else: label = QLabel()

    if desc: label.setToolTip(desc)

    # Attach the pixmap to the label and return
    label.setPixmap(mask); label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label

############################################################################

# Helper function to add a list of layouts and widgets to a main layout
def add_to_layout(
        layout: QHBoxLayout | QVBoxLayout, items: list[QWidget | QLayout], size_ratios: list[int] = None,
        stretches: list[int] = None, alignment: Qt.AlignmentFlag = Qt.AlignmentFlag(0)
):
    # Fallback parameters
    if stretches is None: stretches = []
    if size_ratios is None: size_ratios = [0] * len(items)

    # Iterate through items and add them to main layout, adding stretches, size ratios and alignments if needed
    for index, item in enumerate(items):
        if index in stretches: layout.addStretch()

        if isinstance(item, QWidget):
            layout.addWidget(item, size_ratios[index], alignment=alignment)
        else: layout.addLayout(item, size_ratios[index])

    if -1 in stretches: layout.addStretch()

############################################################################

