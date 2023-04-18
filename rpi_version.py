import pyaudio
import struct
import math
import sys
from PIL import Image, ImageDraw
from rgbmatrix import RGBMatrix, RGBMatrixOptions

# Define the FFT function
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
    t = [math.e ** (-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
    # Combine the even and odd parts of the array
    return [even[k] + t[k] for k in range(n // 2)] + [even[k] - t[k] for k in range(n // 2)]

# Set up the LED matrix options
options = RGBMatrixOptions()
options.rows = 32
options.cols = 32
options.chain_length = 1
options.parallel = 1
options.hardware_mapping = 'regular'

matrix = RGBMatrix(options=options)

# Set up the audio stream
p = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

# Define the function to draw the bar graph on the LED matrix
def draw_bar_graph(fft_data):
    image = Image.new("RGB", (32, 32))
    draw = ImageDraw.Draw(image)
    num_bars = 32
    bar_width = 1

    # normalize the FFT data
    fft_data = [int(x) for x in fft_data]
    fft_data = [abs(x) for x in fft_data]
    fft_data = fft_data[0:512]
    fft_data = [sum(fft_data[i:i+16])/16 for i in range(0,len(fft_data),16)]
    fft_data = [int(x/2000*31) for x in fft_data]

    for i in range(num_bars):
        bar_height = fft_data[i]
        x1 = i * bar_width
        y1 = 31 - bar_height
        x2 = x1 + bar_width
        y2 = 31
        draw.rectangle((x1, y1, x2, y2), fill=(255, 0, 0))

    matrix.SetImage(image.convert('RGB'))

while True:
    try:
        # Read the audio data from the stream
        data = stream.read(CHUNK)
        data = struct.unpack(str(2 * CHUNK) + 'B', data)

        # Perform the FFT on the audio data
        fft_data = fft(data)

        # Draw the bar graph on the LED matrix
        draw_bar_graph(fft_data)

    except KeyboardInterrupt:
        # Clean up the audio stream and PyAudio object
        stream.stop_stream()
        stream.close()
        p.terminate()
        # Clear the LED matrix before exiting
        matrix.Clear()
        sys.exit(0)

