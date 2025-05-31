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







# s21=1
# x21.append(fGHz)
# y21r.append(s21)

# l21.set_data(x21, y21r)
# d21.set_data(x21, y21r)

# ax21.relim(); 
# ax21.autoscale_view(scalex=False)

# fig.show()

# for i in range(1000):
#     x21.append(fGHz+i*100)
#     s21 = 1 + np.random.random()
#     y21r.append(s21)

#     l21.set_data(x21, y21r)
#     d21.set_data(x21, y21r)
#     i += 1

#     time.sleep(1)
#     print(x21)

class VNA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot test")
        cw = QWidget()
        self.setCentralWidget(cw)
        v = QVBoxLayout(cw)
        self.fig, self.ax21 = plt.subplots(1,1, figsize=(12,9))


        self.x21, self.y21r = [], []

        self.l21, = self.ax21.plot([], [], 'b-', lw=1.2, label='S21 smoothed')
        self.d21, = self.ax21.plot([], [], 'ro', ms=4,   label='S21 raw')

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


    def _update(self):
        fGHz=0.4
        
        s21 = 1 + np.random.random()
        try:
            self.y21r[self.point_index] = s21

        except IndexError:
            self.y21r.append(s21)
            self.x21.append(fGHz+self.point_index*0.1)

        self.l21.set_data(self.x21, self.y21r)
        self.d21.set_data(self.x21, self.y21r)
        self.ax21.relim(); 
        self.ax21.autoscale_view(scalex=False)

        self.canvas.draw_idle()
        self.point_index  += 1
        if self.point_index%20 == 0:
            self.point_index = 0
  

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
