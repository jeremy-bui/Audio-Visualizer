import pyaudio
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
import sys
import time


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
        self.fft_y = np.zeros(int(CHUNK / 2))
        self.curve = self.graphWidget.plot(self.fft_x[:int(CHUNK/2/2)], self.fft_y[:int(CHUNK/2/2)], pen='y')

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
        fft_data = np.fft.fft(data) / len(data)
        fft_data = np.abs(fft_data[:int(len(fft_data) / 2)]) * 2

        # Cut off frequencies above 10kHz
        cutoff_index = np.where(self.fft_x >= 10000)[0][0]
        self.curve.setData(self.fft_x[:cutoff_index], fft_data[:cutoff_index])

        # Adjust the y-axis range to reflect the maximum amplitude threshold
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

