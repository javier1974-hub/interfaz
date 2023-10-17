import numpy as np
from scipy.io import wavfile
import pyqtgraph as pg
import scipy as sp


path_file = './training-a/'
file_extention = '.wav'
path_annotation = './annotations/hand_corrected/training-a_StateAns/'
annotation_file_extention = '_StateAns.mat'
prefix = 'a'
can_files = 409

for i in range(1,can_files+1):
    name = prefix + str(i).zfill(4)
    #print(name)
    file = name + '.wav'
    #print(file)

    a = sp.io.loadmat(path_annotation + name + annotation_file_extention)

# hago decimacion x2 tanto la senial como la posicione de las marcas, esta  muestreado a 2000Hz
    ann=[]

    for i in range(len(a['state_ans'])):
        ann.append([int(a['state_ans'][i][0][0][0]/2), a['state_ans'][i][1][0][0][0]])


    samplerate, data_2k = wavfile.read(path_file + name + file_extention)
    data_1k = sp.signal.decimate(data_2k,2)

    data_1k = data_1k.astype(float)



#armo las mascaras

    anotaciones = np.zeros(len(data_1k))

    for i in range(len(ann)-1):
        if (ann[i][1] == 'diastole'):
            anotaciones[ann[i][0]:ann[(i+1)][0]] = 0
        if (ann[i][1] == 'S1'):
            anotaciones[ann[i][0]:ann[(i + 1)][0]] = 1
        if (ann[i][1] == 'systole'):
            anotaciones[ann[i][0]:ann[(i + 1)][0]] = 2
        if (ann[i][1] == 'S2'):
            anotaciones[ann[i][0]:ann[(i + 1)][0]] = 3

    # primero tengo que armar tramos de 1000 muestras y sacarlos a los archivos que sean
    # con un nombre indicando numero de tramo. A su vez podria considerar overlapping para
    # hacer data augmentation

    print(len(data_1k))
    data_1k.tofile(name + '.csv',  sep=",")
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
