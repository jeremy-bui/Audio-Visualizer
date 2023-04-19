from PIL import Image
from PIL import ImageDraw
import pyaudio
import numpy as np
from rgbmatrix import RGBMatrix, RGBMatrixOptions


'''
Computes the DFT (discrete Fourier transform) of the input data. The DFT is a transform that converts a time-domain signal into a frequency-domain representation
'''
def fft(data):
    n = len(data)
    # If the length n is less than or equal to 1, the function returns the input data as is, as there is nothing to transform.
    if n == 1:
        return data
    # Otherwise, the input data is divided into two lists: even and odd, containing the even and odd-indexed elements of data respectively.
    '''
    Splitting the input into even and odd parts simplifies the FFT algorithm because it reduces the number of required calculations. 
    After the split, each of the even and odd parts can be processed recursively using the same FFT algorithm. This reduces the 
    problem size by a factor of 2 at each recursion level, which leads to a significant reduction in the number of calculations required to compute the FFT.
    '''
    even = fft(data[0::2])
    odd = fft(data[1::2])
    '''
    The fft function then calculates the "twiddle factors" using the formula np.exp(-2j * np.pi * k / n), where k is the index of the sample 
    and n is the total length of the input array. It multiplies each odd-indexed sample by the corresponding twiddle factor and stores the results in a temporary array t.
    Odd-indexed samples are multiplied by the twiddle factors because they represent the imaginary part of the frequency components. The even-indexed samples represent the 
    real part of the frequency components and do not require a phase shift.
    '''
    # Combine the even and odd parts of the array
    for k in range(n // 2):
        t = np.exp(-2j * np.pi * k / n) * odd[k]
        data[k] = even[k] + t
        data[k + n // 2] = even[k] - t
    return data


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

#Set up Image that will be drawn
image = Image.new("RGB", (32, 32), color='black')

# Declare Draw instance
draw = ImageDraw.Draw(image)

while True:
 
    # Set up the FFT
    fft_x = np.linspace(0, RATE/2, int(CHUNK / 2)) # Returns an evenly spaced array of 512 nums from 0 to 22050

    # Set the maximum amplitude threshold in volts
    max_amplitude_threshold = 0.1

    data = stream.read(1024, exception_on_overflow = False)
    data = np.frombuffer(data, dtype=np.int16)

    '''
    Because we are using a 16 bit integer, the range is currently between [-32768, 32767]
    We scale this down to [-1, 1]
    '''
    data = data / 32768.0

    # Apply the limiter to the data
    data = np.clip(data, -max_amplitude_threshold, max_amplitude_threshold)

    # Perform the FFT
    # Scaling, so divide by len(data) to have the values be a representation with respect to the CHUNK
    fft_data = np.array(fft(data)) / len(data) 
    # The output of the FFT is symmetric around the midpoint of the array, so we can discard one half of the array without losing any information.
    # We then multiply by 2 to account for the half that was discarded since the magnitudes there matter.
    # So we only consider the data of the first half up to the Nyquist Frequency.
    fft_data = np.abs(fft_data[:int(len(fft_data) / 2)]) * 2
    
    
    # Calculate the amplitude for each frequency range
    '''
    The variable bar_freqs is an array of 33 evenly spaced frequencies between 0 Hz and 10,000 Hz.
    For each of the 32 ranges defined by bar_freqs, the function calculates the corresponding FFT indices by scaling the frequency range 
    with respect to the sample rate of 44,100 Hz and multiplying by the length of the chunk size.
    '''
    bar_data = np.zeros(32)
    bar_freqs = np.linspace(0, 10000, 33)
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
        # Added by start_index to account for the index of the full len(fft_data) because we are only analyzing
        # a subset at the moment.
        max_index = np.argmax(fft_data[start_index:end_index]) + start_index

        # Calculate the amplitude for the current range and multiply by a constant factor
        amplitude_factor = 10 # Experiment with different factors to see what works best
        bar_data[i] = fft_data[max_index] * amplitude_factor
    
        y_value = int(bar_data[i] * 32)
        
        
        draw.line(((31 - i), 0, 31 - i, y_value), fill='white')
        
    matrix.Clear()
    matrix.SetImage(image, 0, 0)
    
    
            
    print(bar_data)

