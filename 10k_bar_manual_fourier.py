import pyaudio
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
import sys
import time
import math

def fft(data):
    n = len(data)
    if n <= 1:
        return data
    even = fft(data[0::2])
    odd = fft(data[1::2])
    t = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
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
        self.timer.start(1)  # Reduce interval to increase update rate

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
        bar_data = np.zeros(32)
        bar_freqs = np.linspace(0, 10000, 33)
        for i in range(32):
            # Find the FFT indices for the current frequency range
            start_index = int(bar_freqs[i] / 44100 * 1024)
            end_index = int(bar_freqs[i + 1] / 44100 * 1024)

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
