
import numpy as np
from scipy.io import wavfile
import pyqtgraph as pg
import scipy as sp
from prettytable import PrettyTable








Diastole_a = np.genfromtxt('intervalos_Diastole_grupo_a.csv', dtype=float, delimiter=',')  # abro el .csv
Diastole_b = np.genfromtxt('intervalos_Diastole_grupo_b.csv', dtype=float, delimiter=',')  # abro el .csv
Diastole_c = np.genfromtxt('intervalos_Diastole_grupo_c.csv', dtype=float, delimiter=',')  # abro el .csv
Diastole_d = np.genfromtxt('intervalos_Diastole_grupo_d.csv', dtype=float, delimiter=',')  # abro el .csv
Diastole_f = np.genfromtxt('intervalos_Diastole_grupo_f.csv', dtype=float, delimiter=',')  # abro el .csv

S1_a = np.genfromtxt('intervalos_S1_grupo_a.csv', dtype=float, delimiter=',')  # abro el .csv
S1_b = np.genfromtxt('intervalos_S1_grupo_b.csv', dtype=float, delimiter=',')  # abro el .csv
S1_c = np.genfromtxt('intervalos_S1_grupo_c.csv', dtype=float, delimiter=',')  # abro el .csv
S1_d = np.genfromtxt('intervalos_S1_grupo_d.csv', dtype=float, delimiter=',')  # abro el .csv
S1_f = np.genfromtxt('intervalos_S1_grupo_f.csv', dtype=float, delimiter=',')  # abro el .csv

Systole_a = np.genfromtxt('intervalos_Systole_grupo_a.csv', dtype=float, delimiter=',')  # abro el .csv
Systole_b = np.genfromtxt('intervalos_Systole_grupo_b.csv', dtype=float, delimiter=',')  # abro el .csv
Systole_c = np.genfromtxt('intervalos_Systole_grupo_c.csv', dtype=float, delimiter=',')  # abro el .csv
Systole_d = np.genfromtxt('intervalos_Systole_grupo_d.csv', dtype=float, delimiter=',')  # abro el .csv
Systole_f = np.genfromtxt('intervalos_Systole_grupo_f.csv', dtype=float, delimiter=',')  # abro el .csv

S2_a = np.genfromtxt('intervalos_S2_grupo_a.csv', dtype=float, delimiter=',')  # abro el .csv
S2_b = np.genfromtxt('intervalos_S2_grupo_b.csv', dtype=float, delimiter=',')  # abro el .csv
S2_c = np.genfromtxt('intervalos_S2_grupo_c.csv', dtype=float, delimiter=',')  # abro el .csv
S2_d = np.genfromtxt('intervalos_S2_grupo_d.csv', dtype=float, delimiter=',')  # abro el .csv
S2_f = np.genfromtxt('intervalos_S2_grupo_f.csv', dtype=float, delimiter=',')  # abro el .csv


Diastole_a=np.append(Diastole_a,Diastole_b)
Diastole_a=np.append(Diastole_a,Diastole_c)
Diastole_a=np.append(Diastole_a,Diastole_d)
Diastole_a=np.append(Diastole_a,Diastole_f)

S1_a=np.append(S1_a,S1_b)
S1_a=np.append(S1_a,S1_c)
S1_a=np.append(S1_a,S1_d)
S1_a=np.append(S1_a,S1_f)

Systole_a=np.append(Systole_a,Systole_b)
Systole_a=np.append(Systole_a,Systole_c)
Systole_a=np.append(Systole_a,Systole_d)
Systole_a=np.append(Systole_a,Systole_f)

S2_a=np.append(S2_a,S2_b)
S2_a=np.append(S2_a,S2_c)
S2_a=np.append(S2_a,S2_d)
S2_a=np.append(S2_a,S2_f)



win = pg.plot()
legend = pg.LegendItem((80,60), offset=(70,20))
legend.setParentItem(win.graphicsItem())

Duracion_diastole = pg.PlotDataItem(Diastole_a,  pen ='w')
win.addItem(Duracion_diastole)
legend.addItem(Duracion_diastole,'Duracion_diastole')

Duracion_S1 = pg.PlotDataItem(S1_a, pen='r')
win.addItem(Duracion_S1)
legend.addItem(Duracion_S1,'Duracion S1')

Duracion_systole = pg.PlotDataItem(Systole_a,  pen ='g')
win.addItem(Duracion_systole)
legend.addItem(Duracion_systole,'Duracion systole')

Duracion_S2 = pg.PlotDataItem(S2_a, pen='c')
win.addItem(Duracion_S2)
legend.addItem(Duracion_S2,'Duracion S2')

diastole_array= np.array(Diastole_a)
diastole_media = np.mean(diastole_array)
diastole_desvio = np.std(diastole_array)

S1_array= np.array(S1_a)
S1_media = np.mean(S1_array)
S1_desvio = np.std(S1_array)

sistole_array= np.array(Systole_a)
sistole_media = np.mean(sistole_array)
sistole_desvio = np.std(sistole_array)

S2_array= np.array(S2_a)
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
