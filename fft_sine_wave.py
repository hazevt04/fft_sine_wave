#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Based on StackOverflow question "Fourier Transform of Sine Waves with Unexpected Results"
# https://stackoverflow.com/questions/36293721/fourier-transform-of-sine-waves-with-unexpected-results

def gen_sines(freq, time_interval, rate, amp=1, debug=False):
  w = 2. * np.pi * freq
  t = np.linspace(0, time_interval, time_interval*rate)
  y = amp*np.sin(w * t)
  if debug:
    print("w is {}".format(w))
    print("time_interval*rate is {}".format(time_interval*rate))
  return y

def gen_signals(secs,Fs, debug=False):
  # frequency, duration, sampling rate, amplitude
  # Frequencies must be in terms of the duration
  # in order for the FFT to only have 1 spike
  f1 = 0.5#1/secs
  f2 = 5#100*secs
  a1=10
  a2=15
  print("Fs is {}".format(Fs))
  print("secs is {}".format(secs))
  print("1/secs is {}".format(1/secs))
  print("f1 is {}".format(f1))
  print("f2 is {}".format(f2))
  print("a1 is {}".format(a1))
  print("a2 is {}".format(a2))
  y1 = gen_sines(f1, secs, Fs, 10, debug)
  y2 = gen_sines(f2, secs, Fs, 15, debug)
  y3 = y1 + y2
  signals = [y1, y2, y3]
  return signals

def plot_signals(signals, fs, secs, debug=False):
  nrSigs = len(signals)
  fig = plt.figure(figsize=(12,9))
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
    ax1.set_ylim([-25,25])

    amps = 2*abs(np.fft.fft(s))/len(s)  # scaled power spectrum
    amps = np.array_split( amps, 2 )[0]#[0:100] # only the first 50 frequencies, arbitrarily chosen
    # this should be close to the amplitude:
    #print 'magnitude of amplitudes: ' + str(sum(amps*amps)**0.5)
    print("magnitude of amplitudes: {}".format(str(sum(amps*amps)**0.5)) )
    freqs=np.arange(0, len(amps), 1)/secs
    ax2 = plt.subplot(nrSigs, cols, pltIdc[1])
    ax2.grid(True)
    ax2.set_title(r"$\frac{2 \cdot fft(s)}{len(s)}$")
    ax2.set_xlabel('freq')
    ax2.set_ylabel('amplitude')
    ax2.set_ylim([-25,25])
    ax2.stem(freqs, amps, use_line_collection=True)
  plt.show()

if __name__ == '__main__':
  secs = 3
  Fs = 44100
  debug = True
  signals = gen_signals(secs, Fs, debug)
  plot_signals(signals, Fs, secs, debug)
