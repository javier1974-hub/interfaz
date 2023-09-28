import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel,
QLineEdit, QCheckBox, QTextEdit, QGridLayout,QPushButton,QFileDialog, QTableWidget, QTableWidgetItem,QProgressBar)
from pyqtgraph import PlotWidget, plot
from PyQt6.QtCore import pyqtSignal, QThread
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
from time import sleep
import os
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import numpy as np
import os
import random
from UNET import *
import itertools


# Create worker thread for running tasks like updating
# the progress bar, renaming photos, displaying information
# in the text edit widget.
class Worker(QThread):
    finished = pyqtSignal(list)
    progress = pyqtSignal(int)
    def __init__(self, filename,data):
        super().__init__()
        self.filename = filename
        self.data = data

    def run(self):
        """Long-running task."""
        i = 0
        data1=[]
        data_int=[]
        chunksize = 100
        info = QFileInfo(self.filename)
        filesize = info.size()
        print(filesize)


        N = int(1024/chunksize)
        for i in range(N):
            data =  np.loadtxt(self.filename, dtype=float, delimiter=',', skiprows = i*chunksize, max_rows = chunksize, usecols=0)
            data1.append(data.tolist())

            self.progress.emit(i)

            print(i*chunksize)
            print(self.data)

        self.data = list(itertools.chain.from_iterable(data1))
        self.finished.emit(self.data)

        # N=int(filesize/chunksize)
        # with open(self.filename, 'rb') as f:
        #     for i in range(N):
        #         self.data+= f.read(chunksize)
        #
        #         self.progress.emit(i)
        #         print(self.data)
        #         #print(type(self.data))
        #         #print(type(self.data[0]))
        #     self.finished.emit()



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initializeUI()

    def initializeUI(self):
        """Set up the application's GUI."""
        self.setMinimumSize(1000, 800)
        self.setWindowTitle('Segmentacion de PCG')
        self.setUpMainWindow()
        self.show()

    def setUpMainWindow(self):

        self.pcg=[]
        self.preds = []
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.label = pg.LabelItem(justify='right')
        self.win.addItem(self.label)
        #self.r = pg.PolyLineROI([(0, 0), (10, 10)])



        self.p1 =self.win.addPlot(row=0, col=0)
        # customize the averaged curve that can be activated from the context menu:
        self.p1.avgPen = pg.mkPen('#FFFFFF')
        self.p1.avgShadowPen = pg.mkPen('#8080DD', width=10)
        #self.p1.addItem(self.r)

        self.p2 = self.win.addPlot(row=1, col=0)


        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this
        # item when doing auto-range calculations.



        self.p2.addItem(self.region, ignoreBounds=True)
        self.p1.setAutoVisible(y=True)



        self.model = UNET(1, 64, 3)


        self.intervalo = QLabel("",self)
        self.coord = QLabel("", self)

        self.button_Model = QPushButton("Load Model",self)
        self.button_Model.clicked.connect(self.buttonModelClicked)
        self.button_File = QPushButton("Load File",self)
        self.button_File.clicked.connect(self.buttonFileClicked)
        3#self.button_File.clicked.connect(self.fileLoad())
        self.button_File.setEnabled(False)
        self.button_Segment = QPushButton("Segment",self)
        self.button_Segment.clicked.connect(self.buttonSegmentClicked)
        self.button_Segment.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)


        #self.table = QTableWidget(4,2)

        self.items_grid = QGridLayout()
        self.items_grid.addWidget(self.win , 0, 1, 5,1)
        self.items_grid.addWidget(self.button_Model, 0, 0, 1,1)
        self.items_grid.addWidget(self.button_File , 1, 0, 1,1)
        self.items_grid.addWidget(self.button_Segment, 2, 0, 1, 1)
        self.items_grid.addWidget(self.intervalo, 3, 0, 1, 1)
        self.items_grid.addWidget(self.coord, 4, 0, 1, 1)
        self.items_grid.addWidget(self.progress_bar, 5, 1, 1, 3)


        self.setLayout(self.items_grid)

    def plot_clicked(self):
        print("clicked!")
        self.intervalo_x.setText('zaraza')
        print(self.mousePoint.x())

    def mouseMoved(self, evt):
        pos = evt
        if self.p1.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            index = int(mousePoint.x())
            if index > 0 and index < len(self.pcg):
                self.label.setText("x=%0.2f, y=%0.2f" % (mousePoint.x(), self.pcg[index]))
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
    def mouseClicked(self,evt):
        #self.region.setZValue(10)
        #minX, maxX = self.region.getRegion()
        #self.p1.setXRange(minX, maxX, padding=0)

        items = self.p1.scene().items(evt.scenePos())
        mousePoint = self.vb.mapSceneToView(evt._scenePos)
        if self.p1.sceneBoundingRect().contains(evt._scenePos):
            mousePoint = self.vb.mapSceneToView(evt._scenePos)
            index = int(mousePoint.x())
            print(int(mousePoint.x()), mousePoint.y())
            arrow = pg.ArrowItem(pos=(index, self.pcg[index]), angle=-90)
            self.p1.addItem(arrow)
            self.coord.setText("x=%0.1f, y=%0.2f" % (mousePoint.x(), self.pcg[index]))
            if index > 0 and index < len(self.pcg):
                self.label.setText("x=%0.1f, y=%0.2f" % (mousePoint.x(), self.pcg[index]))
        #pass

    def update(self):
        self.region.setZValue(10)
        minX, maxX = self.region.getRegion()
        self.p1.setXRange(minX, maxX, padding=0)
        self.intervalo.setText("Intervalo = %0.1f" % (maxX - minX))
        #self.coord.setText("x=%0.1f, y=%0.2f" % (mousePoint.x(), self.pcg[index]))


    def updateRegion(self,window, viewRange):
        rgn = viewRange[0]
        self.region.setRegion(rgn)

    def updateProgressBar(self, value):
        self.progress_bar.setValue(value)

    def buttonModelClicked(self, evt):

         model_path, ok = QFileDialog.getOpenFileName(self,"Open File", "","Torch model (*.pth) ")


         #modelo = UNET(1, 64, 3)
         self.model.load_state_dict(torch.load(model_path))
         self.model.eval()

         self.button_File.setEnabled(True)


    def buttonFileClicked(self):

        #self.pcg = self.fileLoad(self.pcg, self.filename)
        self.file_name, ok = QFileDialog.getOpenFileName(self,"Open File", "","csv (*.csv) ")


        self.worker = Worker(self.file_name, self.pcg)

        self.worker.progress.connect(self.updateProgressBar)
        self.worker.start()

        #print("antes de finished")
        self.worker.finished.connect(self.graficar)
        #print("despues de finished")
        #pass


    def graficar(self,data):

        self.pcg = data
        self.time = np.arange(0, len(self.pcg), 1, dtype=np.float32)

        self.p1.plot(self.pcg, pen="r",symbol='o',symbolSize=5 ,symbolBrush="r")
        self.p1.showGrid(x=True, y=True, alpha=0.3)
        #self.p1.plot(data2, pen="g")

        p2d = self.p2.plot(self.pcg, pen="w")
        # bound the LinearRegionItem to the plotted data
        self.region.setClipItem(p2d)

        self.region.sigRegionChanged.connect(self.update)
        self.p1.sigRangeChanged.connect(self.updateRegion)

        self.region.setRegion([0, 1024])

        # cross hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.p1.addItem(self.vLine, ignoreBounds=True)
        self.p1.addItem(self.hLine, ignoreBounds=True)

        self.vb = self.p1.vb
        self.p1.scene().sigMouseMoved.connect(self.mouseMoved)
        self.p1.scene().sigMouseClicked.connect(self.mouseClicked)


        self.button_Segment.setEnabled(True)

    def buttonSegmentClicked(self):

        self.pcg = torch.from_numpy(self.pcg)
        self.pcg = self.pcg.unsqueeze(0)

        torch.transpose(self.pcg, 0, 1)
        pcg_max = self.pcg.max().item()  # normaliza el pcg ya que van de -1  a 1
        pcg_min = self.pcg.min().item()
        self.pcg = (self.pcg - pcg_min) / (pcg_max - pcg_min)  # ahora queda entre 0 y 1
        self.pcg = self.pcg[None, :]

        #print(self.pcg)
        self.pcg = self.pcg.to(device, dtype=torch.float32)  # pongo seniales en gpu
        self.model = self.model.to(device)  # mando el modelo a GPU
        with torch.no_grad():
            scores = self.model(self.pcg)  # calculo scores (tienen dos canales con la pribabilidad que pertenezca a una u otra clase)
            self.preds = torch.argmax(scores,dim=1).float()  # calcula predicciones (como se queda con el mayor en la dimension de canales, da la mascara)

        self.pcg = self.pcg.squeeze(1).cpu().numpy()

        self.pcg = np.transpose(self.pcg)
        self.pcg = np.squeeze(self.pcg)

        self.preds = self.preds.cpu().numpy()
        self.preds = np.transpose(self.preds)
        self.preds = np.squeeze(self.preds)

        #self.graphWidget.clear()
        self.time = np.arange(0, len(self.pcg), 1, dtype=np.float32)



        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.p1.addItem(self.vLine, ignoreBounds=True)
        self.p1.addItem(self.hLine, ignoreBounds=True)

        #self.p1.plot(self.time, self.pcg)
        cm = pg.ColorMap([0.0,0.5, 1.0], ['r', 'b','g'])
        pen = cm.getPen(span=(0.95, 1.05), width=1, orientation='vertical')

        self.p1.plot(self.time, self.preds,pen=pen)
        #self.p1.plot(self.time, self.pcg, pen=pen)

        x1 = np.where(self.preds == 0)
        x2 = np.where(self.preds == 1)
        x3 = np.where(self.preds == 2)


        # self.v2a = self.win.addViewBox(row=1, col=0, lockAspect=True)
        # self.r2a = pg.PolyLineROI([[x2[0].min(), 0], [x2[0].max(), 10], [10, 30], [30, 10]], closed=True)
        # self.v2a.addItem(self.r2a)
        # self.r2b = pg.PolyLineROI([[0, -20], [10, -10], [10, -30]], closed=False)
        # self.v2a.addItem(self.r2b)
        # self.v2a.disableAutoRange('xy')
        # self.v2a.autoRange()


        print("primer uno")
        print(np.shape(x1))
        print(x1[0].min())
        print(x1[0].max())
        print(x2[0].min())
        print(x2[0].max())
        print(x3[0].min())
        print(x3[0].max())





if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())