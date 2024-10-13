import numpy as np
from scipy.io import wavfile
import pyqtgraph as pg
import scipy as sp
from prettytable import PrettyTable
import pandas as pd
import os

prefix = 'a'
inicio = 1
can_files = 409
path_file = './Database/training-' + prefix + '/'
file_extention = '.wav'
path_annotation = './Database/annotations/hand_corrected/training-' + prefix + '_StateAns/'
annotation_file_extention = '_StateAns.mat'
Chunk_Size = 1024

Diastole = []
S1 = []
Systole = []
S2 = []
Diastole_senial = []
Diastole_senial1 = []
Systole_senial = []
S1_senial = []
S2_senial = []
Lista_Std_Var = []
Diastole_Systole = []
List_SNR = []


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    if sd > 0:
        return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))
    else:
        return np.nan




SNR = pd.DataFrame(columns=['senial', 'SNR', 'SNR_dB'])

for i in range(inicio, can_files + 1):
    name = prefix + str(i).zfill(4)  # 5 para el grupó e sino son 4
    file = name + '.wav'

    if os.path.isfile(path_annotation + name + annotation_file_extention):
        a = sp.io.loadmat(path_annotation + name + annotation_file_extention)

        # ann = []
        #
        # for i in range(len(a['state_ans'])):
        #     ann.append([int(a['state_ans'][i][0][0][0]), a['state_ans'][i][1][0][0][0]])

        samplerate, data_2k = wavfile.read(path_file + name + file_extention)
        data_2k_list = data_2k.tolist()
        # for i in range(len(ann) - 1):
        #     if (ann[i][1] == 'diastole'):
        #         Diastole.append(ann[i + 1][0] - ann[i][0])
        #         Diastole_senial.extend(data_2k_list[ann[i][0]:ann[i + 1][0]])
        #     if (ann[i][1] == 'S1'):
        #         S1.append(ann[i + 1][0] - ann[i][0])
        #         S1_senial.extend(data_2k_list[ann[i][0]:ann[i + 1][0]])
        #     if (ann[i][1] == 'systole'):
        #         Systole.append(ann[i + 1][0] - ann[i][0])
        #         Systole_senial.extend(data_2k_list[ann[i][0]:ann[i + 1][0]])
        #     if (ann[i][1] == 'S2'):
        #         S2.append(ann[i + 1][0] - ann[i][0])
        #         S2_senial.extend(data_2k_list[ann[i][0]:ann[i + 1][0]])


        SNR = signaltonoise(data_2k)
        SNR_dB = signaltonoise_dB(data_2k)

        List_SNR.append([name, SNR, SNR_dB])




SNR = pd.DataFrame(List_SNR, columns=['senial', 'SNR', 'SNR_dB'])

filename = 'SNR_Grupo_' + prefix + '.csv'

SNR.to_csv(filename, sep='\t')



# win = pg.plot()
# legend = pg.LegendItem((80, 60), offset=(70, 20))
# legend.setParentItem(win.graphicsItem())
#
# Duracion_diastole = pg.PlotDataItem(Diastole, pen='w')
# win.addItem(Duracion_diastole)
# legend.addItem(Duracion_diastole, 'Duracion_diastole')
#
# Duracion_S1 = pg.PlotDataItem(S1, pen='r')
# win.addItem(Duracion_S1)
# legend.addItem(Duracion_S1, 'Duracion S1')
#
# Duracion_systole = pg.PlotDataItem(Systole, pen='g')
# win.addItem(Duracion_systole)
# legend.addItem(Duracion_systole, 'Duracion systole')
#
# Duracion_S2 = pg.PlotDataItem(S2, pen='c')
# win.addItem(Duracion_S2)
# legend.addItem(Duracion_S2, 'Duracion S2')

#
# if __name__ == "__main__":
#     pg.exec()
