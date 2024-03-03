import pyaudio
import wave
import queue
import threading
import numpy as np
import signal
from led_ring import LEDRing
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
from pixel_ring import apa102
from pixel_ring import pixel_ring
from gpiozero import LED

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 8
RESPEAKER_WIDTH = 2
# run get_index.py to get respeaker device index
RESPEAKER_INDEX = 1
CHUNK = 1024
MAX_DIST = 0.097 # meters
SPEED_SOUND = 343.2 # meters/sec
#adjusted this a bit based on delay readings
MAX_DELAY = 0.00029 #MAX_DIST/SPEED_SOUND
NUM_LED = 12
BRIGHTNESS = 10
ORDER = "rbg"

class MicArray:
    def __init__(self, rate=RESPEAKER_RATE, channels=RESPEAKER_CHANNELS, chunk_size=CHUNK, device_index = RESPEAKER_INDEX):
        # PyAudio instance
        self.p = pyaudio.PyAudio()
        # FIFO queue instance
        self.queue = queue.Queue()
        # thread event to track callback thread
        self.quit_event = threading.Event()
        #respeaker hardware settings/info
        self.channels = channels
        self.sample_rate = rate
        self.chunk_size = chunk_size
        #open respeaker device (all 8 channels)
        self.stream = self.p.open(
            input=True,
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback,
            input_device_index=device_index
        )
        
    def _callback(self, in_data, frame_count, time_info, status):
        # PyAudio calls callback function whenever buffer has data or program needs it
        # PyAudio gives the callback function audio data with in_data
        # put data on queue for later processing
        self.queue.put(in_data)
        return None, pyaudio.paContinue
    
    def read_chunk(self):
        # yield chunks as long as callback thread is not set
        while not self.quit_event.is_set():
            # chunk is a byte list
            # all 8 channels captured at the same time
            # length of data is CHUNK_SIZE*CHANNELS*bytes_per_int
            # chunks recorded by callback function are stored in a queue
            chunk = self.queue.get()
            # break loop when a null chunk is recieved
            if not chunk:
                break
            yield chunk

    def one_chunk(self):
        #data is list of ints stored in memory as bytes ex) x01
        #for a 16 bit integer, 2 bytes are required
        #length of data buffer depends on int width
        #all 8 channels captured at the same time
        #length of data is CHUNK_SIZE*CHANNELS*bytes_per_int
        chunk = self.stream.read(CHUNK)
        
        return chunk
    
    # need to update to run in callback mode
    def record_time(self, record_seconds):
        print("* recording")
        
        buffer = []
        # record time in terms of number of chunks is RESPEAKER_RATE/CHUNK*record_seconds
        for i in range(0, int(RESPEAKER_RATE / CHUNK * record_seconds)):
            # gather a chunk and append it to the total input buffer
            chunk = self.read_chunk();
            buffer.append(chunk)
        buffer = b''.join(buffer)
        
        print("* done recording")
        # return full buffer of appended chunks
        return buffer

    def direction_rms(self, chunk):
        RMS_values = []
        # get amplitude of all 8 channels
        for x in range(0,8):
            # calculate channel RMS value
            amplitude = chunk[x::8]
            RMS = np.sqrt(np.sum(np.square(amplitude))/len(amplitude))
            RMS_values.append(RMS)
        maxidx = np.argmax(RMS_values)
        # could add check to require certain RMS value for activation
        # each LED index represents 60 degrees
        return maxidx*60
    
    def gcc_phat(self, refsig, sig, interp=16):
        nfft = len(refsig)+len(sig)-1
        
        REFSIG = rfft(refsig, nfft)
        SIG = rfft(sig, nfft)
        
        R = REFSIG*np.conj(SIG)
        #R = R/np.abs(R)
        gcc = ifftshift(irfft(R, interp*nfft))
        
        sample_delay = interp*CHUNK-np.argmax(abs(gcc))+1
        
        time_delay = sample_delay/float(interp*RESPEAKER_RATE)
        
        return time_delay, gcc
    
    def direction_gcc(self, chunk):
        #mic pairs: 1-4, 2-5, 3-6
        tau = [0, 0, 0]
        theta = [0, 0, 0]
        
        mic_groups = 3
        #Calculate time delay and broadside angle for all mic pairs
        for i in range(mic_groups):
            #calculate time delay
            tau[i], _ = self.gcc_phat(chunk[0+i::8], chunk[3+i::8])
            if(np.abs(tau[i])>MAX_DELAY):
                tau[i] = MAX_DELAY
            #calculate broadside angle
            theta[i] = np.arcsin(tau[i] / MAX_DELAY) * 180 / np.pi
        #mic_pair = np.argmin(np.abs(tau))
        #theta = np.arcsin(tau[mic_pair]/MAX_DELAY)*180/np.pi
        #broadside angle points perpendicular to the mic pairing
        #measure angle relative to first mic (90-theta) and adjust for measured mic pair
        #theta = (90 - theta)+(mic_pair*60)
            
        #find mic_pair with smallest delay
        min_index = np.argmin(np.abs(tau))
        #convert angle... needs work
        if (min_index != 0 and theta[min_index - 1] >= 0) or (min_index == 0 and theta[mic_groups - 1] < 0):
            theta = (theta[min_index] + 360) % 360
        else:
            theta = (180 - theta[min_index])

        theta = ((theta + 120 + min_index * 60)+180) % 360
        
        return theta
    
    def close(self):
        self.quit_event.set()
        self.queue.put('')
        self.stream.close()
        self.p.terminate()

def main():
    mic_array = MicArray()
    strip = apa102.APA102(num_led=NUM_LED, global_brightness=BRIGHTNESS, order=ORDER)
    
    # hijack keyboard interupt to close out additional allocations
    def signal_handler(sig, num):
        print('Quit')
        mic_array.close()
        strip.clear_strip()

    signal.signal(signal.SIGINT, signal_handler)
    # read_chunk yields a chunk of audio from the input queue
    # read_chunk is iterable as it will yield a new chunk from the queue on every call
    # new chunks are added to the queue in a seperate thread as audio comes in
    # for loop interations changes as chunks are added, runs as long as audio comes in
    last_led = 0
    for chunk in mic_array.read_chunk():
        # convert that chunk from bytes to ints
        # normalize to [-1,1]
        chunk = np.frombuffer(chunk, np.int16)/32767
        angle = mic_array.direction_gcc(chunk)
        #angle = mic_array.direction_rms(chunk)
        print(angle)
        #light proper LED
        #only change state when LED index changes from last iteration
        led_idx = (round(angle/30)-2)%12
        if(last_led != led_idx):
            strip.clear_strip()
            strip.set_pixel_rgb(led_idx, 0xFF0000, bright_percent = 1)
            strip.show()
        last_led = led_idx
        
    
if __name__ == '__main__':
    main()
        
