from mic_array import MicArray
from led_ring import LEDRing
import numpy as np
import matplotlib.pyplot as plt
import time

mic_array = MicArray()
led_ring = LEDRing()

chunk = mic_array.one_chunk()

#samples = np.frombuffer(chunk, dtype=np.int16)

legend = []
chunkies = []
for x in range(0,8):
    amplitude = np.frombuffer(chunk, np.int16)[x::8]/32767
    chunkies.append(np.sum(np.square(amplitude)))
    plt.plot(amplitude)
    plt.ylabel("normalized voltage")
    plt.xlabel("sample number")
    plt.ylim(-.15,.15)
    val = "Mic: " + str(x)
    legend.append(val)
plt.legend(legend)

maxval = np.argmax(chunkies)
print("the max microphone is mic:" + str(maxval))
plt.show()


#amplitude = np.frombuffer(chunk, np.int16)/32767
#angle = mic_array.direction_mag(amplitude)
#print(angle)

#led_ring.light_led(angle)

while True:
    chunk = mic_array.one_chunk()
    chunk = np.frombuffer(chunk, np.int16)/32767
    angle = mic_array.direction_rms(chunk)
    print(angle)
