import numpy as np
from scipy.io import wavfile
import pyqtgraph as pg
import scipy as sp
from scipy.stats import kurtosis

prefix = 'a'
inicio = 2
can_files = 2
path_file = './Database/training-'+ prefix +'/'
file_extention = '.wav'
path_annotation = './Database/annotations/hand_corrected/training-'+ prefix +'_StateAns/'
annotation_file_extention = '_StateAns.mat'
Chunk_Size = 1024


for i in range(inicio,can_files+1):
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
            anotaciones[ann[i][0]:ann[(i + 1)][0]] = 0
        if (ann[i][1] == 'S2'):
            anotaciones[ann[i][0]:ann[(i + 1)][0]] = 2

    # primero tengo que armar tramos de 1000 muestras y sacarlos a los archivos que sean
    # con un nombre indicando numero de tramo. A su vez podria considerar overlapping para
    # hacer data augmentation

    data_1k_array = np.asarray(data_1k)
    k = kurtosis(data_1k)
    print(k)

    window_kurt = 15
    # con 30 ya no se ve
    kurt = []

    for i in range(1, (len(data_1k) - window_kurt)):
        kurt.append(kurtosis(data_1k_array[i:(i + window_kurt)]))



    win = pg.plot()

    Kurt = pg.PlotDataItem(kurt,  pen ='m')
    win.addItem(Kurt)

    senial = pg.PlotDataItem(data_1k*0.001,  pen ='w')
    win.addItem(senial)

    marcas = pg.PlotDataItem(anotaciones*1, pen='r')
    win.addItem(marcas)

    text = pg.TextItem(name,color='g')
    win.addItem(text)
    text.setPos(20,50)


if __name__ == "__main__":
    pg.exec()