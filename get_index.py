import pyaudio

pyaudio_instance = pyaudio.PyAudio()

device_count = pyaudio_instance.get_device_count()
print("NUmber of Audio Devices: ", device_count)
for i in range(device_count):
            #dev is a dict data type
            dev = pyaudio_instance.get_device_info_by_index(i)
            name = dev['name']#.encode('utf-8')
            print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
