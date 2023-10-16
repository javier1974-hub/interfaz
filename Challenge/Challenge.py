import scipy.io
from scipy.io import wavfile
import pyqtgraph as pg



a = scipy.io.loadmat('./annotations/hand_corrected/training-a_StateAns/a0001_StateAns.mat')
print(a)
print(a['state_ans'][0] )   # primera fila del diccionario
print(a['state_ans'][0][0])   # primer elemento de la primera fila del diccionario es la posicion de la marca
print(a['state_ans'][0][1])   # segundo elemento de la primera fila del diccionario sistole, diastolers, S2 o S2


ann=[]

for i in range(len(a['state_ans'])):
    ann.append([a['state_ans'][i][0][0][0], a['state_ans'][i][1][0][0][0]])


samplerate, data = wavfile.read('./training-a/a0001.wav')

win = pg.plot()

print(samplerate)
senial = pg.PlotDataItem(data,  pen ='w')
win.addItem(senial)

marcas = pg.PlotDataItem(data+5000)

win.addItem(marcas)


if __name__ == "__main__":
    pg.exec()
