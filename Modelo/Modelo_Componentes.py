import numpy as np
from scipy.io import wavfile
import pyqtgraph as pg
import scipy as sp
import scipy.signal as sig
from scipy.stats import kurtosis
import emd

def nco(fcw, sr):
    phase =  0
    phase_result = []
    for fcw_samp in fcw:
        ph_step = 2*np.pi* fcw_samp * 1/sr
        phase += ph_step
        phase_result.append(phase)
    return np.cos(phase_result)


sr = 1000
ts = 1.0/sr
duration = 2
t = np.arange(0,duration,ts)
freq1 =4.20
freq2 = 10
t_change = 1
fcw = np.zeros_like(t)
fcw[t<=t_change] = freq1
fcw[t>t_change] = freq2
result = nco(fcw, sr)


#pg.plot(result)




def nco(frec, amp, fs):
    # fcw: array-like container of frequency control values, one for each sample
    # acw: array-like container of amplitude control values, one for each sample
    # sr: sample rate
    phase =  0
    phase_result = []
    for frec_samp in frec:
        ph_step = 2*np.pi* frec_samp * 1/fs
        phase += ph_step
        phase_result.append(phase)
    return amp * np.cos(phase_result)

# frec1   Hz
# frec2
# frec3
# Dt1 : Duracion en tiempo del intervalo de frec1   seg
# Dt2 : Duracion en tiempo del intervalo de frec2
# Dt3 : Duracion en tiempo del intervalo de frec3
# fs : frecuencia de muestreo de S1   Hz
# T : duracion total de S1 medido en tiempo (s)

def S(frec1, frec2,frec3,Dt1, Dt2, Dt3, A1, A2, A3, fs):

    T = Dt1 + Dt2 + Dt3
    N1 = Dt1 * fs
    N2 = N1 + Dt2 * fs
    N3 = N2 + Dt3 * fs

    frec_vec = np.zeros(T*fs)

    phase =  0
    Sonido = []

    for i in range(len(frec_vec)):
        if (i <= N1):
            ph_step = 2 * np.pi * frec1 * 1 / fs
            phase += ph_step
            Sonido.append(A1* np.cos(phase))
        elif ((i> N1) and (i <= N2)):
            ph_step = 2 * np.pi * frec2 * 1 / fs
            phase += ph_step
            Sonido.append(A2 * np.cos(phase))
        elif ((i>N2) and (i<=N3)):
            ph_step = 2 * np.pi * frec3 * 1 / fs
            phase += ph_step
            Sonido.append(A3 * np.cos(phase))
    return Sonido

S1=S(1,4,2,3,1,3, 1,1,1,1000)
Window = sig.windows.gaussian(500, 10)
S1_2 = sig.filtfilt(Window, np.sum(Window), S1)

Window1 = sig.windows.gaussian(len(S1), len(S1)/15)
S1_3 = Window1 * S1

S1_3_array = np.asarray(S1_3)
k = kurtosis(S1_3)
print(k)


imf = emd.sift.sift(S1_3_array)
print(imf.shape)
#emd.plotting.plot_imfs(imf)

window_kurt = 10
# con 30 ya no se ve
kurt = []

for i in range(1,(len(S1_3)-window_kurt)):
    kurt.append( kurtosis(S1_3_array[i:(i + window_kurt)]))


# sr = 2000
# ts = 1.0/sr
# duration = 3
# t = np.arange(0,duration,ts)
# freq1 =4.20
# freq2 = 6.66
# freq3 = 5
#
# amp1 = 0.05
# amp2 = 1
# amp3 = 0.05
#
# t_change0 = 0.5
# t_change1 = 1
# t_change2 = 2
# t_change3 = 2.5
#

# nsamps = int(taper_dur * sr) * 2
# coeff = sig.windows.gaussian(nsamps, 500)
# acw2 = sig.filtfilt(coeff, np.sum(coeff), acw)
# result = nco(fcw, acw2, sr)
# result1 = nco(fcw, acw, sr)



win = pg.plot()
legend = pg.LegendItem((80,60), offset=(70,20))
legend.setParentItem(win.graphicsItem())


IMF_0 = pg.PlotDataItem(imf[:,0],  pen ='w')
win.addItem(IMF_0)
legend.addItem(IMF_0,'IMF 0')

IMF_1 = pg.PlotDataItem(imf[:,1],  pen ='y')
win.addItem(IMF_1)
legend.addItem(IMF_1,'IMF 1')

IMF_DIFF_1 = pg.PlotDataItem(np.abs(np.diff(imf[:,1]))*10000,  pen ='g')
win.addItem(IMF_DIFF_1)
legend.addItem(IMF_DIFF_1,'IMF_DIFF_1')


IMF_2 = pg.PlotDataItem(imf[:,2],  pen ='c')
win.addItem(IMF_2)
legend.addItem(IMF_2,'IMF 2')



s1_3 = pg.PlotDataItem(S1_3,  pen ='r')
win.addItem(s1_3)
legend.addItem(s1_3,'Mult.  Gauss')

#window = pg.PlotDataItem(Window,  pen ='c')
#win.addItem(window)
#legend.addItem(window,'window')

imf_1= imf[:,1]

# en el paper usa 140 muestras de ventana con señal muestreada a 44100Hz
# eso son 3.2 ms
# entonce con muestreo de 1000Hz son 3-4 muestras

window_kurt = 20
# con 30 ya no se ve
kurt = []

for i in range(1,(len(imf_1)-window_kurt)):
    kurt.append( kurtosis(imf_1[i:(i + window_kurt)]))

kurt_array = np.asarray(kurt)
kurt_abs_diff= np.abs(np.diff(kurt_array))*1e-3


Kurt = pg.PlotDataItem(kurt,  pen ='m')
win.addItem(Kurt)
legend.addItem(Kurt,'kurtosis')

Kurt_abs_diff = pg.PlotDataItem(kurt_abs_diff,  pen ='c')
win.addItem(Kurt_abs_diff)
legend.addItem(Kurt_abs_diff,'Abs Diff Kurt imf 1')
#
# Acw2 = pg.PlotDataItem(acw2,  pen ='g')
# win.addItem(Acw2)
# legend.addItem(Acw2,'Gauss')
#
# Coeff = pg.PlotDataItem(coeff,  pen ='c')
# win.addItem(Coeff)
# legend.addItem(Coeff,'Coeff')


if __name__ == "__main__":
    pg.exec()
