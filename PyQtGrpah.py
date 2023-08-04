import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel,
QLineEdit, QCheckBox, QTextEdit, QGridLayout,QPushButton,QFileDialog)
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import numpy as np
import os
import random
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split

# Modelos NN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# bloque con constantes, ya que usa kernel_size=3  , stride=1 y poadding=1 en todos los casos
# padding =same es para que queden las señales de salida del mismo tamaño que las de entrada
# ahora cada vez que se use un bloque de convolucion solo se le pasa cantidad de canales de entrada y cantidad de canales de salida
# lo demas ya queda implicito
class Conv_3_k(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv1d(channels_in, channels_out, kernel_size=3, stride=1,
                               padding='same')  # de movida va conv 1d

    def forward(self, x):
        return self.conv1(x)


# Se hace el bloque de dos convoluciones una detras de la otra

class Double_Conv(nn.Module):
    '''
    Double convolution block for U-Net
    '''

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv_3_k(channels_in, channels_out),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),

            Conv_3_k(channels_out, channels_out),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


# aca se hace el maxpoolin para bajar y volver a hacer la doble convolucion

class Down_Conv(nn.Module):
    '''
    Down convolution part
    '''

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool1d(2, 2),
            Double_Conv(channels_in, channels_out)
        )

    def forward(self, x):
        return self.encoder(x)


# aca se hace la interpolacion para subir y volver a hacer la doble convolucion

class Up_Conv(nn.Module):
    '''
    Up convolution part
    '''

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            # nn.MaxPool1d(2, 2),
            nn.Upsample(scale_factor=2, mode='linear'),  # interpola y luego hace convolucion de 1x1
            nn.Conv1d(channels_in, channels_in // 2, kernel_size=1, stride=1)
        )
        self.decoder = Double_Conv(channels_in, channels_out)

    def forward(self, x1, x2):
        '''
        x1 - upsampled volume
        x2 - volume from down sample to concatenate
        '''
        x1 = self.upsample_layer(x1)
        x = torch.cat([x2, x1], dim=1)  # concantena a lo largo de la dimension de los canales
        return self.decoder(x)


# aca se hace el modelo


class UNET(nn.Module):
    '''
    UNET model
    '''

    def __init__(self, channels_in, channels, num_classes):
        super().__init__()
        self.first_conv = Double_Conv(channels_in, channels)  # 64, 1024
        self.down_conv1 = Down_Conv(channels, 2 * channels)  # 128, 512
        self.down_conv2 = Down_Conv(2 * channels, 4 * channels)  # 256, 256
        self.down_conv3 = Down_Conv(4 * channels, 8 * channels)  # 512, 128

        self.middle_conv = Down_Conv(8 * channels, 16 * channels)  # 1024, 64

        self.up_conv1 = Up_Conv(16 * channels, 8 * channels)
        self.up_conv2 = Up_Conv(8 * channels, 4 * channels)
        self.up_conv3 = Up_Conv(4 * channels, 2 * channels)
        self.up_conv4 = Up_Conv(2 * channels, channels)

        self.last_conv = nn.Conv1d(channels, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)

        x5 = self.middle_conv(x4)

        u1 = self.up_conv1(x5, x4)
        u2 = self.up_conv2(u1, x3)
        u3 = self.up_conv3(u2, x2)
        u4 = self.up_conv4(u3, x1)
        n = self.last_conv(u4)

        return n


def test():
    x = torch.randn((8, 1, 1024))
    model = UNET(1, 64, 3)
    return model(x)


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
        self.preds = []
        self.graphWidget = MyPlotWidget()
        self.model = UNET(1, 64, 3)


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
        self.button_File.setEnabled(False)
        self.button_Segment = QPushButton("Segment",self)
        self.button_Segment.clicked.connect(self.buttonSegmentClicked)
        self.button_Segment.setEnabled(False)


        self.items_grid = QGridLayout()
        self.items_grid.addWidget(self.graphWidget , 0, 1, 5,1)
        self.items_grid.addWidget(self.button_Model, 0, 0, 1,1)
        self.items_grid.addWidget(self.button_File , 1, 0, 1,1)
        self.items_grid.addWidget(self.coord_x, 2, 0, 1, 1)
        self.items_grid.addWidget(self.button_Segment, 2, 0, 1, 1)

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
         model_path, ok = QFileDialog.getOpenFileName(self,"Open File", "","Torch model (*.pth) ")

         # Cargo el modelo previamente guardado

         #modelo = UNET(1, 64, 3)
         self.model.load_state_dict(torch.load(model_path))
         self.model.eval()

         self.button_File.setEnabled(True)

    def buttonFileClicked(self):

        self.graphWidget.clear()
        self.file_name, ok = QFileDialog.getOpenFileName(self,"Open File", "","csv (*.csv) ")
        self.pcg = np.genfromtxt(self.file_name,dtype =float, delimiter=',')
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

        #print(np.transpose(self.pcg).shape)
        #print(np.squeeze(np.transpose(self.pcg)).shape)
        self.pcg = np.transpose(self.pcg)
        self.pcg = np.squeeze(self.pcg)

        self.preds = self.preds.cpu().numpy()
        #plot_salida(pcg, preds)
        self.preds = np.transpose(self.preds)
        self.preds = np.squeeze(self.preds)

        self.graphWidget.clear()
        self.time = np.arange(0, len(self.pcg), 1, dtype=np.float32)

        # Set Range
        self.graphWidget.setXRange(0, len(self.pcg), padding=0)
        self.graphWidget.setYRange(self.preds.min(), self.preds.max(), padding=0)
        #print(self.pcg)
        self.graphWidget.plot(self.time, self.pcg)
        self.graphWidget.plot(self.time, self.preds)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())