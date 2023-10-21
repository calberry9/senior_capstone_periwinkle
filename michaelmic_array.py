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
            output=True,
            format=pyaudio.paInt16,
            #format=pyaudio.get_format_from_width(RESPEAKER_WIDTH),
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback,
            input_device_index=device_index
        )
        
        self.stream.start_stream()
        
    def _callback(self, in_data, frame_count, time_info, status):
        # PyAudio calls callback function whenever buffer has data or program needs it
        # PyAudio gives the callback function audio data with in_data
        # put data on queue for later processing
        self.queue.put(in_data)
        return None, pyaudio.paContinue
    
    def read_chunk(self):
        # yield chunks as long as callback thread is not set
  
        # chunk is a byte list
        # all 8 channels captured at the same time
        # length of data is CHUNK_SIZE*CHANNELS*bytes_per_int
        # chunks recorded by callback function are stored in a queue
        
        x=0;
        while(x<10000):
            x = x+1
        print("length of queue" + str(self.queue.qsize()))
        print('\n')
        chunk = self.queue.get()
        # break loop when a null chunk is recieved
      
           
        return chunk
    
    #def read_chunk(self):
        #data is list of ints stored in memory as bytes ex) x01
        #for a 16 bit integer, 2 bytes are required
        #length of data buffer depends on int width
        #all 8 channels captured at the same time
        #length of data is CHUNK_SIZE*CHANNELS*bytes_per_int
        #chunk = self.stream.read(CHUNK)
        #print(len(data))
        #print(data
        
        #chunk = np.frombuffer(chunk, np.int16)
        
        #return chunk
    
    # need to update to run in callback mode
    def record_time(self, record_seconds):
        print("* recording")
        
        buffer = []
        # record time in terms of number of chunks is RESPEAKER_RATE/CHUNK*record_seconds
        for i in range(0, int(RESPEAKER_RATE / CHUNK * record_seconds)):
            # gather a chunk and append it to the total input buffer
            chunk = self.read_chunk();
            print(chunk)
            if chunk:
                buffer.append(chunk)
                print("made it to write\n")
                #self.stream.write(chunk)
        buffer = b''.join(buffer)
        
        print("* done recording")
        # return full buffer of appended chunks
        return buffer

    def direction_mag(self, chunk):
        RMS_values = []
        # get amplitude of all 8 channels
        for x in range(0,8):
            # calculate channel RMS value
            RMS = np.sqrt(np.sum(np.square(chunk))/len(chunk))
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
        tau = [0, 0, 0]
        theta = [0, 0, 0]
        
        mic_groups = 3
        for i in range(mic_groups):
            tau[i], _ = self.gcc_phat(chunk[0+i::8], chunk[3+i::8])
            theta[i] = np.arcsin(tau[i]/MAX_DELAY)*180/np.pi
        maxidx = np.argmax(tau)
        
        #if tau[maxidx] >= 0:
            #angle = maxidx*60+90-theta[maxidx]
        #else:
            #angle = 180-maxidx*60-theta[maxidx]
        
        return maxidx*60
    
    def close(self):
        self.quit_event.set()
        self.queue.put('')
        self.stream.close()
        self.p.terminate()
        
    
if __name__ == '__main__':
    mic_array = MicArray()
    led_ring = LEDRing()
    
    # hijack keyboard interupt to close out additional allocations
    def signal_handler(sig, num):
        print('Quit')
        mic_array.close()
        led_ring.clear()
    mic_array.record_time(3)

    signal.signal(signal.SIGINT, signal_handler)
    # read_chunk yields a chunk of audio from the input queue
    # read_chunk is iterable as it will yield a new chunk from the queue on every call
    # new chunks are added to the queue in a seperate thread as audio comes in
    # for loop interations changes as chunks are added, runs as long as audio comes in
    for chunk in mic_array.read_chunk():
        led_ring.clear()
        # convert that chunk from bytes to ints
        # normalize to [-1,1]
        chunk = np.frombuffer(chunk, np.int16)/32767
        angle = mic_array.direction_gcc(chunk)
        #print(angle)
        #angle = mic_array.direction_mag(chunk)
        #print(angle)
        led_ring.light_led(angle)
        
