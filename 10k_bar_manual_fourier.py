import pyaudio
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
import sys
import time

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

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Create a PlotWidget and set it as the central widget of the window
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        # Set up PyAudio
        CHUNK = 1024 
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)

        # Set up the FFT
        self.fft_x = np.linspace(0, RATE/2, int(CHUNK / 2))

        #Bar graph setup in PyQt5 that will eventually be replaced by drawing in the LED matrix
        self.bar_x = np.arange(32)  # x-values for bar graph 
        self.bar_y = np.zeros(32)  # y-values for bar graph
        self.bar_item = pg.BarGraphItem(x=self.bar_x, height=self.bar_y, width=0.8, brush='y')
        self.graphWidget.addItem(self.bar_item)

        # Set up the y-axis range
        self.y_range = 0.1  # Set to a small fraction of maximum amplitude threshold
        self.graphWidget.setYRange(0, self.y_range)

        # Start the update timer
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)  # Reduce interval to increase update rate

        # Initialize the framerate counter
        self.frame_count = 0
        self.start_time = time.time()

    def update(self):
        # Set the maximum amplitude threshold in volts
        max_amplitude_threshold = 0.1

        data = self.stream.read(1024)
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
        bar_freqs = np.linspace(0, 10000, 33)
        '''
        If we only had an array of size 32, we would not be able to include the highest frequency value in the range because it would not have a corresponding array index. 
        Therefore, we use an array of size 33 to include all 32 frequency values and an extra value to represent the upper bound of the frequency range. This way, each bar has a 
        corresponding frequency value and we can display all 32 bars with their respective heights.
        '''
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

        # Update the bar graph
        self.bar_item.setOpts(height=bar_data)

        # Update the y-axis range to reflect the maximum amplitude threshold
        self.y_range = max_amplitude_threshold * 1.1  # Increase y-axis range slightly
        self.graphWidget.setYRange(0, self.y_range)

        # Update the framerate counter
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:
            framerate = self.frame_count / elapsed_time
            print("Framerate: {:.2f}".format(framerate))
            self.frame_count = 0
            self.start_time = time.time()



def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
