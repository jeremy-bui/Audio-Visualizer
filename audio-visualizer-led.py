

from PIL import Image
from PIL import ImageDraw
import time
from rgbmatrix import RGBMatrix, RGBMatrixOptions

import pyaudio
import numpy as np


'''
Splitting the input into even and odd parts simplifies the FFT algorithm because it reduces the number of required calculations. 
After the split, each of the even and odd parts can be processed recursively using the same FFT algorithm. This reduces the 
problem size by a factor of 2 at each recursion level, which leads to a significant reduction in the number of calculations required to compute the FFT.
'''
def fft(data):
    n = len(data)
    # If the length n is less than or equal to 1, the function returns the input data as is, as there is nothing to transform.
    if n <= 1:
        return data
    # Otherwise, the input data is divided into two lists: even and odd, containing the even and odd-indexed elements of data respectively.
    even = fft(data[0::2])
    odd = fft(data[1::2])
    '''
    The fft function then calculates the "twiddle factors" using the formula np.exp(-2j * np.pi * k / n), where k is the index of the sample 
    and n is the total length of the input array. It multiplies each odd-indexed sample by the corresponding twiddle factor and stores the results in a temporary array t.
    '''
    t = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    # Combine the even and odd parts of the array
    return [even[k] + t[k] for k in range(n // 2)] + [even[k] - t[k] for k in range(n // 2)]

# Set up PyAudio
CHUNK = 1024 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK,
                  input_device_index=1)

# Configuration for the matrix
options = RGBMatrixOptions()
options.rows = 32
options.chain_length = 1
options.parallel = 1
options.hardware_mapping = 'regular'  

matrix = RGBMatrix(options = options)
#matrix.brightness = 10

image = Image.new("RGB", (32, 32), color='black')  

draw = ImageDraw.Draw(image)

while True:
    

    
    '''
    draw.rectangle((31, 31, 31, 31), fill='white')
    matrix.SetImage(image, 0, 0)
    
    time.sleep(0.5)
    draw.rectangle((31, 31, 31, 31), fill='black')
    matrix.SetImage(image, 0, 0)
    
    time.sleep(0.5)
    matrix.Clear()
    matrix.SetImage(image, 0, 0)
    '''
    
    # Set up the FFT
    fft_x = np.linspace(0, RATE/2, int(CHUNK / 2))

    # Set the maximum amplitude threshold in volts
    max_amplitude_threshold = 0.1

    data = stream.read(1024, exception_on_overflow = False)
    data = np.frombuffer(data, dtype=np.int16)

    # Scale the data to range from -1 to 1
    data = data / 32768.0

    # Apply the limiter to the data
    data = np.clip(data, -max_amplitude_threshold, max_amplitude_threshold)

    # Perform the FFT
    fft_data = np.array(fft(data)) / len(data)
    fft_data = np.abs(fft_data[:int(len(fft_data) / 2)]) * 2
    
    
    # Calculate the amplitude for each frequency range
    '''
    The variable bar_freqs is an array of 33 evenly spaced frequencies between 0 Hz and 10,000 Hz.
    For each of the 32 ranges defined by bar_freqs, the function calculates the corresponding FFT indices by scaling the frequency range 
    with respect to the sample rate of 44,100 Hz and multiplying by the length of the chunk size.
    '''
    bar_data = np.zeros(32)
    bar_freqs = np.linspace(0, 1100, 33)
    '''
    If we only had an array of size 32, we would not be able to include the highest frequency value in the range because it would not have a corresponding array index. 
    Therefore, we use an array of size 33 to include all 32 frequency values and an extra value to represent the upper bound of the frequency range. This way, each bar has a 
    corresponding frequency value and we can display all 32 bars with their respective heights.
    '''
    
    draw.rectangle((0, 0, 31, 31), fill=(0,0,0))
    for i in range(32):
        # Find the FFT indices for the current frequency range
        start_index = int(bar_freqs[i] / 44100 * 1024)
        end_index = int(bar_freqs[i + 1] / 44100 * 1024)
        '''
        if you input a 5,000 Hz tone, the code would calculate the amplitude of the frequency range from 4,878 Hz to 5,303 Hz, which is the range that includes the 5,000 Hz tone. 
        The start index for this range would be int(4878/44100*1024) = 113 and the end index would be int(5303/44100*1024) = 123. The code would then find the FFT index with the 
        highest amplitude in this range and multiply it by a constant factor to get the amplitude for this frequency range.
        '''

        # Find the FFT index with the highest amplitude in the current range
        max_index = np.argmax(fft_data[start_index:end_index]) + start_index

        # Calculate the amplitude for the current range and multiply by a constant factor
        amplitude_factor = 10 # Experiment with different factors to see what works best
        bar_data[i] = fft_data[max_index] * amplitude_factor
    
        y_value = int(bar_data[i] * 32)
        if y_value > 5:
            #matrix.brightness = 128
            #print('got brighter')
            draw.line(((31 - i), 0, 31 - i, y_value), fill='red')
        else:
            #matrix.brightness = 10
            #print('got dimmer')
            draw.line(((31 - i), 0, 31 - i, y_value), fill='white')
        
        #draw.line(((31 - i), 0, 31 - i, y_value), fill='white')
        
    matrix.Clear()
    matrix.SetImage(image, 0, 0)
    
    
            
    #print(bar_data)

