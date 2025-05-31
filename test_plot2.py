import matplotlib.pyplot as plt
import matplotlib.animation as anim
import traceback
import time
import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QFrame, QProgressDialog, QMessageBox, QCheckBox)
from PySide6.QtCore import QTimer
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT, FigureCanvasQTAgg as FigureCanvas)

MIN_FREQ, MAX_FREQ, SWEEP_INIT_DELAY = 0.3e9, 6e9, 2.0

class VNA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot test")
        cw = QWidget()
        self.setCentralWidget(cw)
        v = QVBoxLayout(cw)
        self.fig, self.ax21 = plt.subplots(1,1, figsize=(12,9))


        self.x21, self.y21r = [], []
        self.x21_freq, self.y21_value = [], []

        self.l21, = self.ax21.plot([], [], 'b-', lw=1.2, label='S21 smoothed1')
        self.d21, = self.ax21.plot([], [], 'ro', ms=4,   label='S21 raw1')

        self.s21_line_continuous, = self.ax21.plot([], [], 'y-', lw=1.2, label='S21 smoothed2')
        self.s21_dots_continuous, = self.ax21.plot([], [], 'go', ms=4,   label='S21 raw2')


        self.ax21.set_title("S21 (dB)")

        self.ax21.set_xlim(MIN_FREQ/1e9, MAX_FREQ/1e9)
        self.ax21.set_ylabel("dB")

        self.ax21.grid(True)


        self.canvas = FigureCanvas(self.fig)
        v.addWidget(self.canvas)
        v.addWidget(NavigationToolbar2QT(self.canvas, self))
        self.point_index = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(1000)
        self.first = True


    def _update(self):
        fGHz=0.4
        if self.first:
            x_data = self.x21
            y_data = self.y21r
            s21_plot_line = self.l21
            s21_plot_dot = self.d21
            print("First plot")
        else:
            x_data = self.x21_freq
            y_data = self.y21_value
            s21_plot_line = self.s21_line_continuous
            s21_plot_dot = self.s21_dots_continuous
            print("Not first plot")

        s21 = 1 + np.random.random()
        try:
            y_data[self.point_index] = s21

        except IndexError:
            y_data.append(s21)
            x_data.append(fGHz+self.point_index*0.1)

        s21_plot_line.set_data(x_data, y_data)
        s21_plot_dot.set_data(x_data, y_data)

        self.ax21.relim(); 
        self.ax21.autoscale_view(scalex=False)

        self.canvas.draw_idle()
        self.point_index  += 1
        if self.point_index%20 == 0:
            self.point_index = 0
            self.first = False
  

if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        win = VNA()
    except Exception:
        traceback.print_exc()
        QMessageBox.critical(None, "Init error", "Failed to start:\n" + traceback.format_exc())
        sys.exit(1)
    win.show()
    sys.exit(app.exec())
