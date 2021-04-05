#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Based on StackOverflow question "Fourier Transform of Sine Waves with Unexpected Results"
# https://stackoverflow.com/questions/36293721/fourier-transform-of-sine-waves-with-unexpected-results

def sine(freq, time_interval, rate, amp=1):
  w = 2. * np.pi * freq
  t = np.linspace(0, time_interval, time_interval*rate)
  y = amp*np.sin(w * t)
  return y

def buildData(secs,Fs):
  # frequency, duration, sampling rate, amplitude
  y1 = sine(0.5, secs, Fs, 10)
  y2 = sine(5, secs, Fs, 15)
  y3 = y1 + y2
  signals = [y1, y2, y3]
  return signals

def showSignals(signals, fs, secs):
  nrSigs = len(signals)
  fig = plt.figure()
  fig.subplots_adjust(hspace=.5)
  for i in range(len(signals)):
    cols=2
    pltIdc = []
    for col in range(1,cols+1):
        pltIdc.append(i*cols+col)
    s = signals[i]
    t = np.arange(0, secs, 1.0/fs)
    ax1 = plt.subplot(nrSigs, cols, pltIdc[0])
    ax1.set_title('signal')
    ax1.set_xlabel('time')
    ax1.set_ylabel('amplitude')
    ax1.plot(t, s)

    amps = 2*abs(np.fft.fft(s))/len(s)  # scaled power spectrum
    amps = np.array_split( amps, 2 )[0][0:50] # only the first 50 frequencies, arbitrarily chosen
    # this should be close to the amplitude:
    #print 'magnitude of amplitudes: ' + str(sum(amps*amps)**0.5)
    print("magnitude of amplitudes: {}".format(str(sum(amps*amps)**0.5)) )
    freqs=np.arange(0, len(amps), 1)/secs
    ax2 = plt.subplot(nrSigs, cols, pltIdc[1])
    ax2.grid(True)
    ax2.set_title(r"$\frac{2 \cdot fft(s)}{len(s)}$")
    ax2.set_xlabel('freq')
    ax2.set_ylabel('amplitude')
    ax2.stem(freqs, amps, use_line_collection=True)
  plt.show()

if __name__ == '__main__':
  secs = 3
  Fs = 44100
  signals = buildData(secs, Fs)
  showSignals(signals, Fs, secs)
