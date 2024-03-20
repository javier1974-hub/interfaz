import numpy as np
from scipy.io import wavfile
import pyqtgraph as pg
import scipy as sp
from scipy.stats import kurtosis
import emd

prefix = 'a'
inicio = 1
can_files = 1
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



    #for i in range(1, (len(data_1k) - window_kurt)):
    #    kurt.append(kurtosis(data_1k_array[i:(i + window_kurt)]))

    imf = emd.sift.sift(data_1k)
    print(imf.shape)

    #k = np.abs(np.diff(imf[:, 1])) * 1000

    imf_2 = imf[:, 2]

    window_kurt = 4
    # con 30 ya no se ve
    kurt = []

    for i in range(1, (len(imf_2) - window_kurt)):
        kurt.append(kurtosis(imf_2[i:(i + window_kurt)]))

    kurt_array = np.asarray(kurt)
    kurt_abs_diff = np.abs(np.diff(kurt_array))*1e1


    win = pg.plot()
    legend = pg.LegendItem((80, 60), offset=(70, 20))
    legend.setParentItem(win.graphicsItem())

    Kurt_abs_diff  = pg.PlotDataItem(kurt_abs_diff,  pen ='m')
    win.addItem(Kurt_abs_diff)
    legend.addItem(Kurt_abs_diff, ' abs de Diff kurtosis de IMF 2')

    #IMF_DIFF_1 = pg.PlotDataItem(np.abs(np.diff(imf[:, 1])) * 0.01, pen='g')
    #win.addItem(IMF_DIFF_1)
    #legend.addItem(IMF_DIFF_1, 'abs de Diff IMF 1')

    senial = pg.PlotDataItem(data_1k*0.001,  pen ='w')
    win.addItem(senial)
    legend.addItem(senial, 'Senial')

    marcas = pg.PlotDataItem(anotaciones*1, pen='r')
    win.addItem(marcas)
    legend.addItem(marcas, 'mascara')

    text = pg.TextItem(name,color='g')
    win.addItem(text)
    text.setPos(20,50)


if __name__ == "__main__":
    pg.exec()