import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel,
QLineEdit, QCheckBox, QTextEdit, QGridLayout,QPushButton,QFileDialog)
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from PyQt6.QtCore import *
from PyQt6.QtGui import *


class MyPlotWidget(pg.PlotWidget):

    sigMouseClicked = pyqtSignal(object) # add our custom signal

    def __init__(self, *args, **kwargs):
        super(MyPlotWidget, self).__init__(*args, **kwargs)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        self.sigMouseClicked.emit(ev)



    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #
    #     # self.scene() is a pyqtgraph.GraphicsScene.GraphicsScene.GraphicsScene
    #     self.scene().sigMouseClicked.connect(self.mouse_clicked)
    #
    #
    # def mouse_clicked(self, mouseClickEvent):
    #     # mouseClickEvent is a pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent
    #     print('clicked plot 0x{:x}, event: {}'.format(id(self), mouseClickEvent))

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initializeUI()

    def initializeUI(self):
        """Set up the application's GUI."""
        self.setMinimumSize(800, 600)
        self.setWindowTitle("Interfaz Prueba")
        self.setUpMainWindow()
        self.show()

    def setUpMainWindow(self):

        self.pcg=[]
        self.graphWidget = MyPlotWidget()


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

        #self.graphWidget.sigMouseClicked.connect(self.plot_clicked)

        #self.graphWidget.plot(hour, temperature, name="Sensor 1",  pen=pen, symbol='+', symbolSize=30, symbolBrush=('b'))
        #self.graphWidget.plot(hour, temperature)
        #self.graphWidget.plot(self.pcg)



        self.coord_x = QLabel("",self)

        self.button_Model = QPushButton("Load Model",self)
        self.button_Model.clicked.connect(self.buttonModelClicked)
        self.button_File = QPushButton("Load File",self)
        self.button_File.clicked.connect(self.buttonFileClicked)
        self.button_File.setEnabled(True)


        self.items_grid = QGridLayout()
        self.items_grid.addWidget(self.graphWidget , 0, 1, 5,1)
        self.items_grid.addWidget(self.button_Model, 0, 0, 1,1)
        self.items_grid.addWidget(self.button_File , 1, 0, 1,1)
        self.items_grid.addWidget(self.coord_x, 2, 0, 1, 1)

        self.setLayout(self.items_grid)

    def plot_clicked(self):
        print("clicked!")
        self.coord_x.setText('zaraza')
        print(self.mousePoint.x())

    def onMouseMoved(self, evt):
        if self.graphWidget.plotItem.vb.mapSceneToView(evt):
            point = self.graphWidget.plotItem.vb.mapSceneToView(evt)
            self.label.setHtml(
                "<p style='color:white'>X： {0} <br> Y: {1}</p>". \
                    format(point.x(), point.y()))

    def mouse_clicked(self, mouseClickEvent):
        # mouseClickEvent is a pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent
        print('clicked plot 0x{:x}, event: {}'.format(id(self), mouseClickEvent))

    def buttonModelClicked(self):
         file_name, ok = QFileDialog.getOpenFileName(self,"Open File", "","Torch model (*.pth) ")
         #self.model = np.genfromtxt(file_name,dtype =float, delimiter=',')
         self.button_File.setEnabled(True)

    def buttonFileClicked(self):
        self.graphWidget.clear()
        file_name, ok = QFileDialog.getOpenFileName(self,"Open File", "","csv (*.csv) ")
        self.pcg = np.genfromtxt(file_name,dtype =float, delimiter=',')
        self.time = np.arange(0,len(self.pcg),1, dtype=np.float32)

        # Set Range
        self.graphWidget.setXRange(0, len(self.pcg), padding=0)
        self.graphWidget.setYRange(self.pcg.min(), self.pcg.max(), padding=0)

        self.graphWidget.plot(self.time,self.pcg)

        self.label = pg.TextItem(text="X: {} \nY: {}".format(0, 0))
        self.graphWidget.addItem(self.label)

        self.setMouseTracking(True)
        self.graphWidget.scene().sigMouseMoved.connect(self.onMouseMoved)
        self.graphWidget.scene().sigMouseClicked.connect(self.mouse_clicked)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())