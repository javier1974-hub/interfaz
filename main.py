# main_window_template.py
# Import necessary modules
import sys
import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import (QApplication, QMainWindow,QFileDialog)
from PyQt6.QtGui import QAction
from vispy.scene import SceneCanvas, visuals, AxisWidget
from vispy.app import use_app
import vispy.plot as vp

IMAGE_SHAPE = (600, 800)  # (height, width)
CANVAS_SIZE = (800, 600)  # (width, height)
NUM_LINE_POINTS = 1024

COLORMAP_CHOICES = ["viridis", "reds", "blues"]
LINE_COLOR_CHOICES = ["black", "red", "blue"]

pcg = np.zeros(1024)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initializeUI()

    def initializeUI(self):
        """Set up the application's GUI."""
        self.setMinimumSize(1000, 900)
        self.setWindowTitle("Main Window Template")
        self.setUpMainWindow()
        self.createActions()
        self.createMenu()
        self.show()

    def setUpMainWindow(self):
        """Create and arrange widgets in the main window."""
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()

        self._controls = Controls()
        main_layout.addWidget(self._controls)
        self._canvas_wrapper = CanvasWrapper()
        main_layout.addWidget(self._canvas_wrapper.canvas.native)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.createActions()


    def createActions(self):
        """Create the application's menu actions."""
        # Create actions for File menu
        self.quit_act = QAction("&Quit")
        self.quit_act.setShortcut("Ctrl+Q")
        self.quit_act.triggered.connect(self.close)

        self.open_act = QAction("&Open")
        self.open_act.setShortcut("Ctrl+O")
        self.open_act.triggered.connect(self.openFile)

        self.save_act = QAction("&Save")
        self.save_act.setShortcut("Ctrl+S")
        self.save_act.triggered.connect(self.saveToFile)

        self._controls.colormap_chooser.currentTextChanged.connect(self._canvas_wrapper.set_image_colormap)
        self._controls.line_color_chooser.currentTextChanged.connect(self._canvas_wrapper.set_line_color)



    def createMenu(self):
        """Create the application's menu bar."""
        self.menuBar().setNativeMenuBar(False)
        # Create file menu and add actions
        file_menu = self.menuBar().addMenu("File")
        file_menu.addSeparator()
        file_menu.addAction(self.open_act)
        file_menu.addAction(self.save_act)
        file_menu.addAction(self.quit_act)

    def openFile(self):
        """Open a text or html file and display its contents
        in the text edit field."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Csv Files (*.csv)")

        if file_name:
            pcg = np.genfromtxt(file_name, dtype=float, delimiter=',')  # abro el .csv

            pos_actual = np.empty((len(pcg), 2), dtype=np.float32)
            pos_actual[:, 0] = np.arange(len(pcg))
            pos_actual[:, 1] = pcg

            self._canvas_wrapper.set_signal(pos_actual)

            print(pcg.shape)
            print(len(pcg))



    def saveToFile(self):
        """If the save button is clicked, display dialog
        asking user if they want to save the text in the text
        edit field to a text or rich text file."""

        #file_name, _ = QFileDialog.getSaveFileName(self, "Save File", " ", "")
        #if file_name.endswith(".txt"):
        #    notepad_text = self.text_edit.toPlainText()
        #with open(file_name, "w") as f:
        #    f.write(notepad_text)

class Controls(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout()
        self.colormap_label = QtWidgets.QLabel("Image Colormap:")
        layout.addWidget(self.colormap_label)
        self.colormap_chooser = QtWidgets.QComboBox()
        self.colormap_chooser.addItems(COLORMAP_CHOICES)
        layout.addWidget(self.colormap_chooser)

        self.line_color_label = QtWidgets.QLabel("Line color:")
        layout.addWidget(self.line_color_label)
        self.line_color_chooser = QtWidgets.QComboBox()
        self.line_color_chooser.addItems(LINE_COLOR_CHOICES)
        layout.addWidget(self.line_color_chooser)

        layout.addStretch(1)
        self.setLayout(layout)


class CanvasWrapper:
    def __init__(self):
        self.canvas = SceneCanvas(size=CANVAS_SIZE)
        self.grid = self.canvas.central_widget.add_grid()


        self.x_axis_top = AxisWidget(axis_label="X Axis Label", orientation='bottom')
        self.x_axis_top.stretch = (1, 0.1)

        self.y_axis_top  = AxisWidget(axis_label="Y Axis Label", orientation='left')
        self.y_axis_top.stretch = (0.1, 1)

        self.grid.add_widget(self.y_axis_top , row=0, col=0)
        self.grid.add_widget(self.x_axis_top , row=1, col=1)


        self.view_top = self.grid.add_view(0, 1, bgcolor='black')


        self.view_top.camera = "panzoom"
        self.x_axis_top.link_view(self.view_top)
        self.y_axis_top.link_view(self.view_top)

        self.view_top.camera.set_range(x=(0, IMAGE_SHAPE[1]), y=(0, IMAGE_SHAPE[0]), margin=0)


        self.x_axis_bot = AxisWidget(axis_label="X Axis Label", orientation='bottom')
        self.x_axis_bot.stretch = (1, 0.1)

        self.y_axis_bot = AxisWidget(axis_label="Y Axis Label", orientation='left')
        self.y_axis_bot.stretch = (0.1, 1)

        self.grid.add_widget(self.y_axis_bot, row=2, col=0)
        self.grid.add_widget(self.x_axis_bot, row=3, col=1)


        self.view_bot = self.grid.add_view(2, 1, bgcolor='black')

#---------------------------------------------------------------------------------
        self.view_bot.camera = "panzoom"

        self.x_axis_bot.link_view(self.view_bot)
        self.y_axis_bot.link_view(self.view_bot)
        self.view_bot.camera.set_range(x=(0, NUM_LINE_POINTS), y=(0, 1))

    def set_image_colormap(self, cmap_name: str):
        print(f"Changing image colormap to {cmap_name}")
        self.image.cmap = cmap_name

    def set_line_color(self, color):
        print(f"Changing line color to {color}")
        self.line.set_data(color=color)

    def set_signal(self,signal):
        self.line=visuals.Line(signal, parent=self.view_bot.scene, color='#c0c0c0')

    def clear_signal(self):
        self.view_bot = self.grid.add_view(2, 1, bgcolor='black')


def _generate_random_image_data(shape, dtype=np.float32):
    rng = np.random.default_rng()
    data = rng.random(shape, dtype=dtype)
    return data

def _generate_random_line_positions(num_points, dtype=np.float32):
    rng = np.random.default_rng()
    pos = np.empty((num_points, 2), dtype=np.float32)
    pos[:, 0] = np.arange(num_points)
    pos[:, 1] = rng.random((num_points,), dtype=dtype)
    return pos

def _generate_line_positions(num_points, dtype=np.float32):
    rng = np.random.default_rng()
    #np.genfromtxt(file_name, dtype=float, delimiter=',')
    pos = np.empty((num_points, 2), dtype=np.float32)
    pos[:, 0] = np.arange(num_points)
    pos[:, 1] = pcg #rng.random((num_points,), dtype=dtype)
    print(pos[:, 0])
    print(pos[:, 1])
    return pos

if __name__ == '__main__':
    app = use_app("pyqt6")
    app.create()
    win = MainWindow()
    win.show()
    app.run()