import numpy as np
import scipy.io
from scipy.io import wavfile
import pyqtgraph as pg
import os


path_file = './training-e/'
file_extention = '.wav'
path_annotation = './annotations/hand_corrected/training-e_StateAns/'
annotation_file_extention = '_StateAns.mat'
prefix = 'e'
can_files = 2141

for i in range(2092,can_files+1):
    name = prefix + str(i).zfill(5)
    #print(name)
    file = name + '.wav'
    #print(file)

    a = scipy.io.loadmat(path_annotation + name + annotation_file_extention)

    ann=[]

    for i in range(len(a['state_ans'])):
        ann.append([a['state_ans'][i][0][0][0], a['state_ans'][i][1][0][0][0]])


    samplerate, data = wavfile.read(path_file + name + file_extention)

    data = data.astype(float)
    data.tofile(name + '.csv',  sep=",")

    anotaciones = np.zeros(len(data))

    for i in range(len(ann)-1):
        if (ann[i][1] == 'diastole'):
            anotaciones[ann[i][0]:ann[(i+1)][0]] = 0
        if (ann[i][1] == 'S1'):
            anotaciones[ann[i][0]:ann[(i + 1)][0]] = 1
        if (ann[i][1] == 'systole'):
            anotaciones[ann[i][0]:ann[(i + 1)][0]] = 2
        if (ann[i][1] == 'S2'):
            anotaciones[ann[i][0]:ann[(i + 1)][0]] = 3

    anotaciones.tofile(name + '_mask.csv',  sep=",")


# win = pg.plot()
# print(samplerate)
# senial = pg.PlotDataItem(data,  pen ='w')
# win.addItem(senial)
#
# marcas = pg.PlotDataItem(anotaciones*1000, pen='r')
# win.addItem(marcas)


if __name__ == "__main__":
    pg.exec()
