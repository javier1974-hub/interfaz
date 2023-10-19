import numpy as np
from scipy.io import wavfile
import pyqtgraph as pg
import scipy as sp

prefix = 'c'
can_files = 31
path_file = './Database/training-'+ prefix +'/'
file_extention = '.wav'
path_annotation = './Database/annotations/hand_corrected/training-'+ prefix +'_StateAns/'
annotation_file_extention = '_StateAns.mat'


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

    #print(len(data_1k))
    len_data = len(data_1k)
    chunks = int(len_data/1000)

    # data_1k_T= np.zeros(1000)
    # anotaciones_T = np.zeros(1000)

    for i in range(chunks):
        data_1k[(i*1000):((i*1000)+1000)].tofile('./train/'+ name + '_'+ str(i+1) +'.csv',  sep=",")
        anotaciones[(i*1000):((i*1000)+1000)].tofile('./train_masks/'+ name + '_'+ str(i+1) + '_mask.csv',  sep=",")


        # data_1k_graph = data_1k[(i*1000):((i*1000)+1000)]
        # anotaciones_graph = anotaciones[(i * 1000):((i * 1000) + 1000)]
        # win = pg.plot()
        # senial = pg.PlotDataItem(data_1k_graph,  pen ='w')
        # win.addItem(senial)
        #
        # marcas = pg.PlotDataItem(anotaciones_graph*1000, pen='r')
        # win.addItem(marcas)



    data_1k_zero_padding = data_1k[(chunks * 1000):len_data]
    anotaciones_zero_padding = anotaciones[(chunks * 1000):len_data]


    data_1k_zero_padding = np.pad(data_1k_zero_padding,(0,(1000-(len_data-chunks*1000))),'constant')
    anotaciones_zero_padding = np.pad(anotaciones_zero_padding, (0,(1000-(len_data-chunks*1000))), 'constant')


    data_1k_zero_padding.tofile('./train/'+name + '_' + str(chunks+ 1) + '.csv', sep=",")
    anotaciones_zero_padding.tofile('./train_masks/'+ name + '_' + str(chunks + 1) + '_mask.csv', sep=",")


if __name__ == "__main__":
    pg.exec()
