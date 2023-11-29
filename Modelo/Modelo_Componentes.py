import numpy as np
from scipy.io import wavfile
import pyqtgraph as pg
import scipy as sp
import scipy.signal as sig

# def nco(fcw, sr):
#     phase =  0
#     phase_result = []
#     for fcw_samp in fcw:
#         ph_step = 2*np.pi* fcw_samp * 1/sr
#         phase += ph_step
#         phase_result.append(phase)
#     return np.cos(phase_result)
#
#
# sr = 22050
# ts = 1.0/sr
# duration = 2
# t = np.arange(0,duration,ts)
# freq1 =4.20
# freq2 = 10
# t_change = 1
# fcw = np.zeros_like(t)
# fcw[t<=t_change] = freq1
# fcw[t>t_change] = freq2
# result = nco(fcw, sr)
#
#
# pg.plot(result)
#



def nco(fcw, acw, sr):
    # fcw: array-like container of frequency control values, one for each sample
    # acw: array-like container of amplitude control values, one for each sample
    # sr: sample rate
    phase =  0
    phase_result = []
    for fcw_samp in fcw:
        ph_step = 2*np.pi* fcw_samp * 1/sr
        phase += ph_step
        phase_result.append(phase)
    return acw * np.cos(phase_result)




sr = 22050
ts = 1.0/sr
duration = 3
t = np.arange(0,duration,ts)
freq1 =4.20
freq2 = 6.66
freq3 = 5

amp1 = 0.1
amp2 = 1
amp3 = 0.1

t_change1 = 1
t_change2 = 2

taper_dur = 0.2
b=10

fcw = np.zeros_like(t)
acw = np.zeros_like(t)
fcw[t<=t_change1] = freq1
fcw[t>t_change1 ] = freq2

fcw[t>t_change2 ] = freq3

acw[t<=t_change1] = amp1
acw[t>t_change1] = amp2
acw[t>t_change2 ] = amp3

nsamps = int(taper_dur * sr) * 2
coeff = sig.windows.gaussian(nsamps, 1000)
acw2 = sig.filtfilt(coeff, np.sum(coeff), acw)
result = nco(fcw, acw2, sr)

result1 = nco(fcw, acw, sr)



win = pg.plot()
legend = pg.LegendItem((80,60), offset=(70,20))
legend.setParentItem(win.graphicsItem())




Result = pg.PlotDataItem(result,  pen ='w')
win.addItem(Result)
legend.addItem(Result,'Filtrado')

Result1 = pg.PlotDataItem(result1,  pen ='r')
win.addItem(Result1)
legend.addItem(Result1,'Sin Filtrar')

Acw2 = pg.PlotDataItem(acw2,  pen ='g')
win.addItem(Acw2)
legend.addItem(Acw2,'Gauss')

Coeff = pg.PlotDataItem(coeff,  pen ='c')
win.addItem(Coeff)
legend.addItem(Coeff,'Coeff')


if __name__ == "__main__":
    pg.exec()
