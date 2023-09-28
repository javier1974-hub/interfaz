import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
import numpy as np

class LinePlot(pg.GraphicsObject):
   def __init__(self, data):
       pg.GraphicsObject.__init__(self)
       pg.setConfigOptions(antialias=True)

       self.data = data
       self.width = 2
       self.pos = (55,108,91,255)
       self.neg = (131,54,70,255)
       self.generatePicture()

   def generatePicture(self):
       self.picture = QtGui.QPicture()
       p = QtGui.QPainter(self.picture)
       for i in range(len(self.data) - 1):
           if data[i] > data[i + 1]:
               p.setPen(pg.mkPen(self.neg, width=self.width))
           else:
               p.setPen(pg.mkPen(self.pos, width=self.width))
           p.drawLine(QtCore.QPointF(i, self.data[i]), QtCore.QPointF(i + 1, self.data[i + 1]))

   def paint(self, p, *args):
       p.drawPicture(0, 0, self.picture)

   def boundingRect(self):
       return QtCore.QRectF(self.picture.boundingRect())




if __name__ == '__main__':
   import sys


data = np.random.normal(size=1000)
item = LinePlot(data)
p1 = pg.plot()
p1.addItem(item)