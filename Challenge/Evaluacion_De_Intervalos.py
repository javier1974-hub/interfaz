import numpy as np
from scipy.io import wavfile
import pyqtgraph as pg
import scipy as sp
from prettytable import PrettyTable

prefix = 'f'
inicio = 1
can_files = 114
path_file = './Database/training-'+ prefix +'/'
file_extention = '.wav'
path_annotation = './Database/annotations/hand_corrected/training-'+ prefix +'_StateAns/'
annotation_file_extention = '_StateAns.mat'
Chunk_Size = 1024

Diastole = []
S1 = []
Systole = []
S2 = []

for i in range(inicio,can_files+1):
    name = prefix + str(i).zfill(4)
    file = name + '.wav'

    a = sp.io.loadmat(path_annotation + name + annotation_file_extention)

# hago decimacion x2 tanto la senial como la posicione de las marcas, esta  muestreado a 2000Hz
    ann=[]

    for i in range(len(a['state_ans'])):
        ann.append([int(a['state_ans'][i][0][0][0]/2), a['state_ans'][i][1][0][0][0]])


    for i in range(len(ann)-1):
        if (ann[i][1] == 'diastole'):
            Diastole.append(ann[i+1][0] - ann[i][0])
        if (ann[i][1] == 'S1'):
            S1.append(ann[i+1][0] - ann[i][0])
        if (ann[i][1] == 'systole'):
            Systole.append(ann[i+1][0] - ann[i][0])
        if (ann[i][1] == 'S2'):
            S2.append(ann[i+1][0] - ann[i][0])

Diastole_array= np.array(Diastole)
Diastole_array.tofile('intervalos_Diastole_grupo_'+ prefix +'.csv',sep=',')
S1_array= np.array(S1)
S1_array.tofile('intervalos_S1_grupo_'+ prefix +'.csv',sep=',')
Systole_array= np.array(Systole)
Systole_array.tofile('intervalos_Systole_grupo_'+ prefix +'.csv',sep=',')
S2_array= np.array(S2)
S2_array.tofile('intervalos_S2_grupo_'+ prefix +'.csv',sep=',')


win = pg.plot()
legend = pg.LegendItem((80,60), offset=(70,20))
legend.setParentItem(win.graphicsItem())

Duracion_diastole = pg.PlotDataItem(Diastole,  pen ='w')
win.addItem(Duracion_diastole)
legend.addItem(Duracion_diastole,'Duracion_diastole')

Duracion_S1 = pg.PlotDataItem(S1, pen='r')
win.addItem(Duracion_S1)
legend.addItem(Duracion_S1,'Duracion S1')

Duracion_systole = pg.PlotDataItem(Systole,  pen ='g')
win.addItem(Duracion_systole)
legend.addItem(Duracion_systole,'Duracion systole')

Duracion_S2 = pg.PlotDataItem(S2, pen='c')
win.addItem(Duracion_S2)
legend.addItem(Duracion_S2,'Duracion S2')

diastole_array= np.array(Diastole)
diastole_media = np.mean(diastole_array)
diastole_desvio = np.std(diastole_array)

S1_array= np.array(S1)
S1_media = np.mean(S1_array)
S1_desvio = np.std(S1_array)

sistole_array= np.array(Systole)
sistole_media = np.mean(sistole_array)
sistole_desvio = np.std(sistole_array)

S2_array= np.array(S2)
S2_media = np.mean(S2_array)
S2_desvio = np.std(S2_array)


t = PrettyTable(['Intervalo','Media', 'Desvio'])
t.add_row(['Diastole', str(diastole_media),str(diastole_desvio) ])
t.add_row(['S1', str(S1_media),str(S1_desvio)])
t.add_row(['Sistole', str(sistole_media),str(sistole_desvio)])
t.add_row(['S2', str(S2_media),str(S2_desvio)])
print(t)



if __name__ == "__main__":
    pg.exec()
