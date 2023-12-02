import numpy as np
from scipy.io import wavfile
import pyqtgraph as pg
import scipy as sp
import scipy.signal as sig

def nco(fcw, sr):
    phase =  0
    phase_result = []
    for fcw_samp in fcw:
        ph_step = 2*np.pi* fcw_samp * 1/sr
        phase += ph_step
        phase_result.append(phase)
    return np.cos(phase_result)


sr = 22050
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

S1=S(5,20,5,2,2,2, 0.2,1,0.1,1000)


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

s1 = pg.PlotDataItem(S1,  pen ='w')
win.addItem(s1)
legend.addItem(s1,'Filtrado')


# Result = pg.PlotDataItem(result,  pen ='w')
# win.addItem(Result)
# legend.addItem(Result,'Filtrado')
#
# Result1 = pg.PlotDataItem(result1,  pen ='r')
# win.addItem(Result1)
# legend.addItem(Result1,'Sin Filtrar')
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
