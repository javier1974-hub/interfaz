from PyQt6.QtWidgets import (QApplication, QWidget, QLabel,
QLineEdit, QCheckBox, QTextEdit, QGridLayout,QPushButton)
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initializeUI()

    def initializeUI(self):
        """Set up the application's GUI."""
        self.setMinimumSize(800, 600)
        self.setWindowTitle("Interfaz Prueba")
        self.setUpMainWindow()
        #self.loadWidgetValuesFromFile()
        self.show()

    def setUpMainWindow(self):

        self.pizza_label = QLabel("Pizza Type: ")

        self.graphWidget = pg.PlotWidget()


        hour = [1,2,3,4,5,6,7,8,9,10]
        temperature = [30,32,34,32,33,31,29,32,35,45]

        #Add Background colour to white
        self.graphWidget.setBackground('k')
        # Add Title
        self.graphWidget.setTitle("Señal", color="w", size="30pt")
        # Add Axis Labels
        styles = {"color": "#fff", "font-size": "20px"}
        self.graphWidget.setLabel("left", "Amp", **styles)
        self.graphWidget.setLabel("bottom", "muestras", **styles)

        #Add legend
        self.graphWidget.addLegend()
        #Add grid
        self.graphWidget.showGrid(x=True, y=True)
        #Set Range
        self.graphWidget.setXRange(0, 10, padding=0)
        self.graphWidget.setYRange(20, 55, padding=0)

        pen = pg.mkPen(color=(255, 0, 0))
        #self.graphWidget.plot(hour, temperature, name="Sensor 1",  pen=pen, symbol='+', symbolSize=30, symbolBrush=('b'))
        self.graphWidget.plot(hour, temperature)


        self.button_Model = QPushButton("Load Model")
        self.button_Model.clicked.connect(self.buttonModelClicked)
        self.button_File = QPushButton("Load File")
        self.button_File.clicked.connect(self.buttonFileClicked)
        self.button_File.setEnabled(False)


        self.items_grid = QGridLayout()
        self.items_grid.addWidget(self.graphWidget , 0, 1, 5,1)
        self.items_grid.addWidget(self.button_Model, 0, 0, 1,1)
        self.items_grid.addWidget(self.button_File , 1, 0, 1,1)

        self.setLayout(self.items_grid)

        def buttonMdelClicked(self):
            pass

        def buttonFileClicked(self):
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())