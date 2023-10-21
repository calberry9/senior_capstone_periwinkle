from mic_array import MicArray
import numpy as np
import matplotlib.pyplot as plt

import time
from scipy.fftpack import rfft, irfft,fftfreq,fft,ifft
from pixel_ring import apa102


from pixel_ring import pixel_ring
from gpiozero import LED

print("is it here")
mic_array = MicArray()

chunk = mic_array.read_chunk()
#print(len(chunk))

#buffer = mic_array.record_time(1)
#print(len(buffer))

#need to convert byte list to a list of ints
#frombuffer interprets data as a 1D array, converts 2 bytes to a 16 bit int
#a is an numpy array of 16 bit integers
#length of a is CHANNELS*CHUNK_SIZE
#samples = np.frombuffer(buffer, dtype=np.int16)
#print(len(samples))

#figure, axis  = plt.subplot(2,2)
legend = []
chunkies = []

ffts = [] 
for x in range(0,8):
    amplitude = np.frombuffer(chunk, np.int16)[x::8]/32767
    
    #ffts.append(fftamplitude)))
    
    
    
    
    plt.plot(fft(amplitude))
    plt.ylabel("normalized voltage")
    plt.xlabel("sample number")
    plt.ylim(-.3,.3)
    val = "Mic: " + str(x)
    legend.append(val)
plt.legend(legend)

#print(amplitude)

#maxval = np.argmax(chunkies)

#print("the max microphone is mic:" + str(maxval))
#
strip = apa102.APA102(num_led=12, global_brightness=10,  

order="rbg")
#strip.set_pixel_rgb(2*(maxval-1), 0xFF0000, bright_percent = 1) # Red


 
strip.show()
#strip.clear_strip()



#print(chunk)
#print(amplitude)
print("made here")

plt.show()

