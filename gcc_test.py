import pyaudio
import wave
import queue
import threading
import numpy as np
import signal
from led_ring import LEDRing
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
 
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 8
RESPEAKER_WIDTH = 2
# run get_index.py to get respeaker device index
RESPEAKER_INDEX = 2
CHUNK = 1024
MAX_DIST = 0.097 # meters
SPEED_SOUND = 343.2 # meters/sec
MAX_DELAY = MAX_DIST/SPEED_SOUND # sec

p = pyaudio.PyAudio()

stream = p.open(
            input=True,
            format=pyaudio.paInt16,
            #format=pyaudio.get_format_from_width(RESPEAKER_WIDTH),
            channels=RESPEAKER_CHANNELS,
            rate=RESPEAKER_RATE,
            frames_per_buffer=CHUNK,
            input_device_index=RESPEAKER_INDEX
        )

chunk = stream.read(CHUNK)

chunk = np.frombuffer(chunk, np.int16)/32767

sig = chunk[0::8]
refsig = chunk[3::8]

nfft = len(refsig)+len(sig)-1
interp = 16

REFSIG = rfft(refsig, nfft)
SIG = rfft(sig, nfft)
        
R = REFSIG*np.conj(SIG)
#R = R/np.abs(R)
gcc = ifftshift(irfft(R, interp*nfft))
        
sample_delay = interp*CHUNK-np.argmax(abs(gcc))

print(sample_delay)

time_delay = sample_delay/float(interp*RESPEAKER_RATE)

theta = np.arcsin(time_delay/MAX_DELAY)*180/np.pi

print(theta)
