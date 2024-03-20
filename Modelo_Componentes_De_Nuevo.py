import numpy as np
from scipy.io import wavfile
import pyqtgraph as pg
import scipy as sp
from matplotlib import pyplot as plt
import math
import scipy.signal as sig
from scipy.stats import kurtosis
import emd

def ending_phase(c, mag):
    angle = math.asin(c[-1] / mag)
    if c[-2] > c[-1]:
        angle = np.pi - angle
    return angle

def next_phase(c, mag):
    ph1 = ending_phase(c[:-1], mag)
    ph2 = ending_phase(c, mag)
    return 2 * ph2 - ph1


N = 1024
t = np.linspace(0, 1, N)

mag = 1.2

freq = 2.2
phase = 0
c1 = mag * np.sin(2 * np.pi * freq * t + phase)

plt.figure(figsize=(8,8))

plt.subplot(3,3,1)
plt.grid()
plt.plot(c1)
plt.ylabel('Tres senoidales')


freq = 5.8
phase = next_phase(c1, mag)
c2 = mag * np.sin(2 * np.pi * freq * t + phase)


plt.subplot(3,3,2)
plt.grid()
plt.plot(c2)

freq = 2.1
phase = next_phase(c2, mag)
c3 = mag * np.sin(2 * np.pi * freq * t + phase)

plt.subplot(3,3,3)
plt.grid()
plt.plot(c3)



c31 = np.concatenate((c1, c2))

#plt.subplot(2,3,4)
#plt.grid()
#plt.plot(c31)

c4 = np.concatenate((c31,c3))


S1 = c4


plt.subplot(3,3,(4,6))
plt.grid()
plt.plot(S1)

plt.ylabel('Senidales concatenadas')

#plt.show()


Window1 = sig.windows.gaussian(len(c4), len(c4)/12)
S1_ventaneado = Window1 * S1

plt.subplot(3,3,(7,9))
plt.grid()
plt.plot(S1_ventaneado)
plt.ylabel('S1 ventaneado')
plt.xlabel('muestras')


S1_ventaneado_array = np.asarray(S1_ventaneado)
k = kurtosis(S1_ventaneado)
print(k)


imf = emd.sift.sift(S1_ventaneado_array)
print(imf.shape)


imf_2= imf[:,2]



window_kurt = 20
# con 30 ya no se ve
kurt = []

for i in range(1,(len(imf_2)-window_kurt)):
    kurt.append( kurtosis(imf_2[i:(i + window_kurt)]))


kurt_array = np.asarray(kurt)
kurt_abs_diff= np.abs(np.diff(kurt_array))*1e-3


#------------------------------------------------
plt.figure(figsize=(8,8))

plt.subplot(6,1,1)
plt.grid()
plt.plot(S1_ventaneado)
plt.ylabel('S1_ventaneado')

plt.subplot(6,1,2)
plt.grid()
plt.plot(imf[:,0])
plt.ylabel('imf 0')

plt.subplot(6,1,3)
plt.grid()
plt.plot(imf[:,1])
plt.ylabel('imf 1')

plt.subplot(6,1,4)
plt.grid()
plt.plot(imf[:,2])
plt.ylabel('imf 2')

plt.subplot(6,1,5)
plt.grid()
plt.plot(imf[:,3])
plt.ylabel('imf 3')

plt.subplot(6,1,6)
plt.grid()
plt.plot(kurt_abs_diff)

plt.ylabel('kurt_abs_diff')
plt.xlabel('muestras')
#-------------------------------------

plt.figure(figsize=(8,8))
plt.subplot(4,1,1)
plt.grid()
plt.plot(S1_ventaneado)
plt.ylabel('S1_ventaneado')
plt.plot(kurt_abs_diff*1e3)

plt.subplot(4,1,2)
plt.grid()
plt.plot(imf[:,2])
plt.ylabel('imf 2')

plt.subplot(4,1,3)
plt.grid()
plt.plot(kurt_array)
plt.ylabel(' kurtosis imf 2')



plt.subplot(4,1,4)
plt.grid()
plt.plot(kurt_abs_diff)

plt.ylabel('kurt_abs_diff')
plt.xlabel('muestras')

plt.show()
