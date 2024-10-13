import numpy as np
from scipy.io import wavfile
import pyqtgraph as pg
import scipy as sp
from prettytable import PrettyTable
from scipy import stats

prefix = 'a'
inicio = 1
can_files = 409
path_file = './Database/training-'+ prefix +'/'
file_extention = '.wav'
path_annotation = './Database/annotations/hand_corrected/training-'+ prefix +'_StateAns/'
annotation_file_extention = '_StateAns.mat'
Chunk_Size = 1024

Diastole = []
S1 = []
Systole = []
S2 = []

pvalue_kstest_Diastole = []
pvalue_kstest_S1 = []
pvalue_kstest_Systole = []
pvalue_kstest_S2 = []

pvalue_jarque_bera_Diastole = []
pvalue_jarque_bera_S1 = []
pvalue_jarque_bera_Systole = []
pvalue_jarque_bera_S2 = []

pvalue_DP_Diastole = [] # DP Dagostino Pearson
pvalue_DP_S1 = []
pvalue_DP_Systole = []
pvalue_DP_S2 = []



for i in range(inicio,can_files+1):
    name = prefix + str(i).zfill(4)
    file = name + '.wav'

    a = sp.io.loadmat(path_annotation + name + annotation_file_extention)
    samplerate, data_2k = wavfile.read(path_file + name + file_extention)
# hago decimacion x2 tanto la senial como la posicione de las marcas, esta  muestreado a 2000Hz
    ann=[]

    for i in range(len(a['state_ans'])):
        ann.append([int(a['state_ans'][i][0][0][0]/2), a['state_ans'][i][1][0][0][0]])


    for i in range(len(ann)-1):
        if (ann[i][1] == 'diastole'):
            Diastole.append(ann[i+1][0] - ann[i][0])
            D, p = stats.kstest(data_2k[ ann[i][0]:ann[i+1][0]], stats.norm.cdf)
            pvalue_kstest_Diastole.append(p)

            res = stats.jarque_bera(data_2k[ann[i][0]:ann[i + 1][0]])
            pvalue_jarque_bera_Diastole.append(res.pvalue)

            res = stats.normaltest(data_2k[ann[i][0]:ann[i + 1][0]])
            pvalue_DP_Diastole.append(res.pvalue)

        if (ann[i][1] == 'S1'):
            S1.append(ann[i+1][0] - ann[i][0])
            D, p = stats.kstest(data_2k[ ann[i][0]:ann[i+1][0]], stats.norm.cdf)
            pvalue_kstest_S1.append(p)

            res = stats.jarque_bera(data_2k[ann[i][0]:ann[i + 1][0]])
            pvalue_jarque_bera_S1.append(res.pvalue)

            res = stats.normaltest(data_2k[ann[i][0]:ann[i + 1][0]])
            pvalue_DP_S1.append(res.pvalue)

        if (ann[i][1] == 'systole'):
            Systole.append(ann[i+1][0] - ann[i][0])
            D, p = stats.kstest(data_2k[ ann[i][0]:ann[i+1][0]], stats.norm.cdf)
            pvalue_kstest_Systole.append(p)

            res = stats.jarque_bera(data_2k[ann[i][0]:ann[i + 1][0]])
            pvalue_jarque_bera_Systole.append(res.pvalue)

            res = stats.normaltest(data_2k[ann[i][0]:ann[i + 1][0]])
            pvalue_DP_Systole.append(res.pvalue)

        if (ann[i][1] == 'S2'):
            S2.append(ann[i+1][0] - ann[i][0])
            D, p = stats.kstest(data_2k[ ann[i][0]:ann[i+1][0]], stats.norm.cdf)
            pvalue_kstest_S2.append(p)

            res = stats.jarque_bera(data_2k[ann[i][0]:ann[i + 1][0]])
            pvalue_jarque_bera_S2.append(res.pvalue)

            res = stats.normaltest(data_2k[ann[i][0]:ann[i + 1][0]])
            pvalue_DP_S2.append(res.pvalue)

Diastole_array= np.array(Diastole)
Diastole_array.tofile('intervalos_Diastole_grupo_'+ prefix +'.csv',sep=',')
S1_array= np.array(S1)
S1_array.tofile('intervalos_S1_grupo_'+ prefix +'.csv',sep=',')
Systole_array= np.array(Systole)
Systole_array.tofile('intervalos_Systole_grupo_'+ prefix +'.csv',sep=',')
S2_array= np.array(S2)
S2_array.tofile('intervalos_S2_grupo_'+ prefix +'.csv',sep=',')

pvalue_kstest_Diastole_array = np.array(pvalue_kstest_Diastole)
pvalue_kstest_Diastole_array.tofile('pValueKstest_Diastole_grupo_'+ prefix +'.csv',sep=',')
pvalue_kstest_S1_array=np.array(pvalue_kstest_S1)
pvalue_kstest_S1_array.tofile('pValueKstest_S1_grupo_'+ prefix +'.csv',sep=',')
pvalue_kstest_Systole_array = np.array(pvalue_kstest_Systole)
pvalue_kstest_Systole_array.tofile('pValueKstest_Systole_grupo_'+ prefix +'.csv',sep=',')
pvalue_kstest_S2_array = np.array(pvalue_kstest_S2)
pvalue_kstest_S2_array.tofile('pValueKstest_S2_grupo_'+ prefix +'.csv',sep=',')


pvalue_jarque_bera_Diastole_array = np.array(pvalue_jarque_bera_Diastole)
pvalue_jarque_bera_Diastole_array.tofile('pValuejarque_bera_Diastole_grupo_'+ prefix +'.csv',sep=',')
pvalue_jarque_bera_S1_array=np.array(pvalue_jarque_bera_S1)
pvalue_jarque_bera_S1_array.tofile('pValuejarque_bera_S1_grupo_'+ prefix +'.csv',sep=',')
pvalue_jarque_bera_Systole_array = np.array(pvalue_jarque_bera_Systole)
pvalue_jarque_bera_Systole_array.tofile('pValuejarque_bera_Systole_grupo_'+ prefix +'.csv',sep=',')
pvalue_jarque_bera_S2_array = np.array(pvalue_jarque_bera_S2)
pvalue_jarque_bera_S2_array.tofile('pValuejarque_bera_S2_grupo_'+ prefix +'.csv',sep=',')


pvalue_DP_Diastole_array = np.array(pvalue_DP_Diastole)
pvalue_DP_Diastole_array.tofile('pValueDP_Diastole_grupo_'+ prefix +'.csv',sep=',')
pvalue_DP_S1_array=np.array(pvalue_DP_S1)
pvalue_DP_S1_array.tofile('pValueDP_S1_grupo_'+ prefix +'.csv',sep=',')
pvalue_DP_Systole_array = np.array(pvalue_DP_Systole)
pvalue_DP_Systole_array.tofile('pValueDP_Systole_grupo_'+ prefix +'.csv',sep=',')
pvalue_DP_S2_array = np.array(pvalue_DP_S2)
pvalue_DP_S2_array.tofile('pValueDP_S2_grupo_'+ prefix +'.csv',sep=',')


win = pg.plot()
legendpValue = pg.LegendItem((80,60), offset=(70,20))
legendpValue.setParentItem(win.graphicsItem())

pValue_kstest_Diastole = pg.PlotDataItem(pvalue_kstest_Diastole,  pen ='w')
win.addItem(pValue_kstest_Diastole)
legendpValue.addItem(pValue_kstest_Diastole,'pValue_kstest_diastole')

pValue_kstest_S1 = pg.PlotDataItem(pvalue_kstest_S1, pen='r')
win.addItem(pValue_kstest_S1)
legendpValue.addItem(pValue_kstest_S1,'pValue_kstest S1')

pValue_kstest_Systole = pg.PlotDataItem(pvalue_kstest_Systole,  pen ='g')
win.addItem(pValue_kstest_Systole)
legendpValue.addItem(pValue_kstest_Systole,'pValue_kstest systole')

pValue_kstest_S2 = pg.PlotDataItem(pvalue_kstest_S2, pen='c')
win.addItem(pValue_kstest_S2)
legendpValue.addItem(pValue_kstest_S2,'pValue_kstest S2')


win1 = pg.plot()
legend = pg.LegendItem((80,60), offset=(70,20))
legend.setParentItem(win1.graphicsItem())

Duracion_diastole = pg.PlotDataItem(Diastole,  pen ='w')
win1.addItem(Duracion_diastole)
legend.addItem(Duracion_diastole,'Duracion_diastole')

Duracion_S1 = pg.PlotDataItem(S1, pen='r')
win1.addItem(Duracion_S1)
legend.addItem(Duracion_S1,'Duracion S1')

Duracion_systole = pg.PlotDataItem(Systole,  pen ='g')
win1.addItem(Duracion_systole)
legend.addItem(Duracion_systole,'Duracion systole')

Duracion_S2 = pg.PlotDataItem(S2, pen='c')
win1.addItem(Duracion_S2)
legend.addItem(Duracion_S2,'Duracion S2')




win2 = pg.plot()
legendjb = pg.LegendItem((80,60), offset=(70,20))
legendjb.setParentItem(win2.graphicsItem())

pValue_jarque_bera_Diastole = pg.PlotDataItem(pvalue_jarque_bera_Diastole,  pen ='w')
win2.addItem(pValue_jarque_bera_Diastole)
legendjb.addItem(pValue_jarque_bera_Diastole,'pValue_jb_diastole')

pValue_jarque_bera_S1 = pg.PlotDataItem(pvalue_jarque_bera_S1, pen='r')
win2.addItem(pValue_jarque_bera_S1)
legendjb.addItem(pValue_jarque_bera_S1,'pValue_jb S1')

pValue_jarque_bera_Systole = pg.PlotDataItem(pvalue_jarque_bera_Systole,  pen ='g')
win2.addItem(pValue_jarque_bera_Systole)
legendjb.addItem(pValue_jarque_bera_Systole,'pValue_jb systole')

pValue_jarque_bera_S2 = pg.PlotDataItem(pvalue_jarque_bera_S2, pen='c')
win2.addItem(pValue_jarque_bera_S2 )
legendjb.addItem(pValue_jarque_bera_S2 ,'pValue_jb S2')


win3 = pg.plot()
legendDP = pg.LegendItem((80,60), offset=(70,20))
legendDP.setParentItem(win3.graphicsItem())

pValue_DP_Diastole = pg.PlotDataItem(pvalue_DP_Diastole,  pen ='w')
win3.addItem(pValue_DP_Diastole)
legendDP.addItem(pValue_DP_Diastole,'pValue_DP_diastole')

pValue_DP_S1 = pg.PlotDataItem(pvalue_DP_S1, pen='r')
win3.addItem(pValue_DP_S1)
legendDP.addItem(pValue_DP_S1,'pValue_DP S1')

pValue_DP_Systole = pg.PlotDataItem(pvalue_DP_Systole,  pen ='g')
win3.addItem(pValue_DP_Systole)
legendDP.addItem(pValue_DP_Systole,'pValue_DP systole')

pValue_DP_S2 = pg.PlotDataItem(pvalue_DP_S2, pen='c')
win3.addItem(pValue_DP_S2 )
legendDP.addItem(pValue_DP_S2 ,'pValue_DP S2')



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
