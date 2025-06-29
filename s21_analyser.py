#!/usr/bin/env python3
"""
PlutoSDR VNA Pro – interactive edition
======================================
• FFT-based FIR lock-in
• Calibration + skip-mask for S11
• Moving-average smoothing **toggle**
• Add/remove markers with left / right mouse clicks
• Clamp S11 to a maximum of 0 dB
"""

# ───────────── imports ─────────────
import sys, os, time, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT, FigureCanvasQTAgg as FigureCanvas)
import adi
from scipy.signal import kaiserord, firwin, lfilter, fftconvolve, oaconvolve
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QFrame, QProgressDialog, QMessageBox, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# ───────────── configuration constants ─────────────
FILTER_MODE, FFT_METHOD = 'fft', 'fftconvolve'
FILT_RIPPLE_DB, FILT_CUTOFF_HZ, FILT_TRANS_WIDTH_HZ = 70, 500, 100
DEFAULT_STEPS, CAL_POINTS = 500, 500
SMOOTH_WIN21, SMOOTH_WIN11 = 7, 5
SDR_URI = "ip:pluto.local"
AD_SAMPLING_FREQUENCY, SAMPLE_BUFFER_SIZE, TX_TONE_FREQ = 8e6, 350000, 543e3
RF_FILTER_BANDWIDTH=4e6
DWELL, CLR_READS, EPS = 0.1, 1, 1e-15
MIN_FREQ, MAX_FREQ, SWEEP_INIT_DELAY = 0.3e9, 6e9, 2.0

# ───────────── hardware initialisation ─────────────
# sdr = adi.Pluto(uri=SDR_URI)
sdr = adi.Pluto("ip:192.168.2.137")
sdr.sample_rate             = int(AD_SAMPLING_FREQUENCY)
sdr.rx_rf_bandwidth         = int(RF_FILTER_BANDWIDTH)
sdr.tx_rf_bandwidth         = int(RF_FILTER_BANDWIDTH)
sdr.rx_buffer_size          = SAMPLE_BUFFER_SIZE
sdr.tx_buffer_size          = SAMPLE_BUFFER_SIZE
sdr.gain_control_mode_chan0 = "manual"
sdr.tx_cyclic_buffer        = True


### TX SIDE ###
sdr.tx_hardwaregain_chan0   = -30
#with -30dB -10dB ext att=-40dB and 20dB att RX, 0dB internal. THere are problems. 5dB attenuation is measured perfectly, but 30dB with lot of oscillations(because this is noise or leakage of TX inside pluto).
# -40-20-30=-90dbm it seems limit of dynamic range for this device not leakage.
# it seems that good option is  to add 10dB ext(necessary to improve impedance matching) att at TX, with -30db internal.
#It gives low internal TX RX leekage and about -40dBm (confirmed with power meter), libreSDR gives about 5dB less power
#Pluto must be supplied from two usb cables without it output power is not stable and changes in time.
#with this configuration and internal rx amp set to 0dB data from IQ is about -71dB refered to 2^12 from ADC.
# Another problem is that with such TX power, open circuit noise reaches -25dB(-117 not calib) at 3GHz, -35dB at 1GHz.(with calibrated through) 
# libreSDR gives similar values  -39dB(-110 not calib) at 1GHz. -24(-109) at 3GHz
# with LNA internal 10dB at 1GHz it gives -36(-107) worst point is 2.13ghz -17dB, and 3.2GHz -14dB, therefore with internal LNA it is not usable due to signal leakage.
#therefore the best idea is to add external PA at TX side.


### RX SIDE ######
#To be below RX IP3 point in full range, the safe input power must be below -18dBm
#when added 10dB(tx side) and 20dB(rx side) attenuators in series to improve impedance
#due to bad impedance match -5dB below 1GHz, at rx port should be 20dB external attenuator added
#without it s21 works really bad, it is bad event with 10dB attenautaor. 30dB attenuator can not be measured due to reflections in cable.
sdr.rx_hardwaregain_chan0   = 0


_t = np.arange(SAMPLE_BUFFER_SIZE) / AD_SAMPLING_FREQUENCY
sdr.tx((np.exp(2j * np.pi * TX_TONE_FREQ * _t) * (2**14)).astype(np.complex64))

# ───────────── FIR filter for lock-in ─────────────
nyq     = AD_SAMPLING_FREQUENCY / 2
N, beta = kaiserord(FILT_RIPPLE_DB, FILT_TRANS_WIDTH_HZ/nyq)
b_fir   = firwin(N, FILT_CUTOFF_HZ/nyq, window=('kaiser', beta))
print(f"Fir taps: {N}")
print(f"Samples buffer size: {SAMPLE_BUFFER_SIZE}")
if SAMPLE_BUFFER_SIZE < N:
    print("WARNING Samples buffer should be higher then filter taps count")

def apply_filter(x):
    if FILTER_MODE == 'direct':
        return lfilter(b_fir, 1.0, x)
    func = fftconvolve if FFT_METHOD == 'fftconvolve' else oaconvolve
    return func(x, b_fir, mode='same')

def lockin(buf: np.ndarray) -> float:
    if np.allclose(buf, 0):
        return 0.0
    t = np.arange(len(buf)) / AD_SAMPLING_FREQUENCY
    y = apply_filter(buf * np.exp(-2j * np.pi * TX_TONE_FREQ * t))
    y = y[N//2:]  # discard FIR transient
    return np.abs(y).mean()

def to_dB(x):
    return 20 * np.log10(np.maximum(x, EPS))

def smooth_trace(y, k):
    if k <= 1 or y.size < k:
        return y
    mask = np.isfinite(y).astype(float)
    win = np.ones(k)
    num = np.convolve(np.nan_to_num(y), win, 'same')
    den = np.convolve(mask, win, 'same')
    out = np.full_like(y, np.nan)
    good = den > 0
    out[good] = num[good] / den[good]
    return out

# ───────────── worker thread ─────────────
class SweepThread(QThread):
    update  = pyqtSignal(float, float)
    reset_signal = pyqtSignal()
    scan_finished = pyqtSignal()
    trigger_start_signal = pyqtSignal()
    error   = pyqtSignal(str)

    def __init__(self, dev):
        super().__init__()
        self.dev       = dev
        self.f0, self.f1, self.n = MIN_FREQ, MAX_FREQ, DEFAULT_STEPS
        self.stop_flag = False
        self.cal21     = None
        self.trigger_start = False
        self.trigger_start_signal.connect(self._trigger_start)


    def _trigger_start(self):
        print("Start trigger emmited")
        self.trigger_start = True

    def set_span(self, f0, f1, n):
        self.f0, self.f1, self.n = f0, f1, n

    def stop(self):
        self.stop_flag = True

    def load_cal21(self, d):
        self.cal21 = d
        print(f"Loaded cals: {d}")


    def _safe_rx(self):
        try:
            return self.dev.rx()
        except Exception as e:
            self.error.emit(f"SDR RX error: {e}")
            self.stop_flag = True
            raise

    def run(self):
        time.sleep(SWEEP_INIT_DELAY)
        self.reset_signal.emit()
        while not self.stop_flag:
            while self.trigger_start is False:
                print("Waiting for start trigger")
                if self.stop_flag:
                    return
                time.sleep(1)
            self.trigger_start = False
            freqs = np.linspace(self.f0, self.f1, self.n)
            for i, f in enumerate(freqs):
                if self.stop_flag or self.trigger_start is True:
                    break
                NUM_R = 4 if f < 1e9 else 1
                try:
                    self.dev.tx_lo = self.dev.rx_lo = int(f)
                except Exception as e:
                    self.error.emit(f"SDR tune error: {e}")
                    self.stop_flag = True
                    break

                time.sleep(DWELL)
                for _ in range(CLR_READS):
                    self._safe_rx()

                acc0 = np.zeros(SAMPLE_BUFFER_SIZE * NUM_R, np.complex64)
                for j in range(NUM_R):
                    r = self._safe_rx()
                    print(f"Freqpoint: {f} IQ:{r}")
                    acc0[j*SAMPLE_BUFFER_SIZE:(j+1)*SAMPLE_BUFFER_SIZE] = (r/2**12)

                s21_amplitude = lockin(acc0)
                s21 = to_dB(s21_amplitude)
                print(f"S21 dB: {s21} raw filtered ADC: {s21_amplitude}")
                offset = None
                if self.cal21 is not None:
                    offset = np.interp(f, self.cal21['freqs'], self.cal21['db'])
                    s21 -= offset
                print(f"S21 calirated: {s21} calib offset: {offset}")


                self.update.emit(f, s21)

            self.scan_finished.emit()
            if not self.stop_flag:
                time.sleep(1)

# ───────────── GUI with toggles + markers ─────────────
class VNA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PlutoSDR VNA Pro – interactive")
        self._build_ui()
        self._init_plot()
        self._spawn_worker()

        if os.path.exists('cal_s21.npz'):
            self.load_s21()

    # --- UI bar + checkboxes ---
    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        box1 = QVBoxLayout(cw)


        top = QFrame()
        top.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)

        h_box1 = QHBoxLayout(top)

        for lbl, attr, val in [
            ("Start (MHz):","le0", int(MIN_FREQ/1e6)),
            ("Stop (MHz):", "le1", int(MAX_FREQ/1e6)),
            ("Steps:",       "leN", DEFAULT_STEPS)
        ]:
            h_box1.addWidget(QLabel(lbl))
            le = QLineEdit(str(val))
            setattr(self, attr, le)
            h_box1.addWidget(le)

        pb = QPushButton("Apply")
        pb.clicked.connect(self.apply_span)
        h_box1.addWidget(pb)

        for txt, fn in [
            ("Cal S21", self.cal_s21),
            ("Load S21", self.load_s21),
        ]:
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            h_box1.addWidget(btn)

        h_box1.addWidget(QLabel("Show:"))
        self.cbSmooth = QCheckBox("Smoothed")
        self.cbSmooth.setChecked(True)
        h_box1.addWidget(self.cbSmooth)
        self.cbRaw = QCheckBox("Raw")
        self.cbRaw.setChecked(True)
        h_box1.addWidget(self.cbRaw)
        self.cbSmooth.stateChanged.connect(self._vis_toggle)
        self.cbRaw.stateChanged.connect(self._vis_toggle)

        self.checkbox_single = QCheckBox("Single scan")
        self.checkbox_single.setChecked(False)
        h_box1.addWidget(self.checkbox_single)

        box1.addWidget(top)

        pb_start = QPushButton("Restart")
        pb_start.clicked.connect(self._start_from_beginning)
        h_box1.addWidget(pb_start)

        self.freq_label = QLabel("f:--")
        h_box1.addWidget(self.freq_label)

        self.fig, self.ax21 = plt.subplots(1,1, figsize=(12,9))
        self.canvas = FigureCanvas(self.fig)
        box1.addWidget(self.canvas)
        box1.addWidget(NavigationToolbar2QT(self.canvas, self))

    def _start_from_beginning(self):
        self._reset()
        self.wk.trigger_start_signal.emit()

    # --- traces + markers storage + titles ---
    def _init_plot(self):
        self.first_plot = True
        self.x21, self.y21r = [], []
        self.x21_freq, self.y21_value = [], []

        self.l21, = self.ax21.plot([], [], 'b-', lw=1.2, label='S21 smoothed')
        self.d21, = self.ax21.plot([], [], 'ro', ms=4,   label='S21 raw')

        self.s21_line_continuous, = self.ax21.plot([], [], 'y-', lw=1.2, label='S21 smoothed2')
        self.s21_dots_continuous, = self.ax21.plot([], [], 'go', ms=4,   label='S21 raw2')

        self.ax21.set_title("S21 (dB)")

        self.ax21.set_xlim(MIN_FREQ/1e9, MAX_FREQ/1e9)
        self.ax21.set_ylabel("dB")
  
        self.ax21.grid(True)

        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.markers = []
        self._vis_toggle()
        self.point_index = 0

    # --- worker startup ---
    def _spawn_worker(self):
        self.wk = SweepThread(sdr)
        self.wk.update.connect(self._update_plot)
        self.wk.reset_signal.connect(self._reset)
        self.wk.error.connect(lambda m: QMessageBox.critical(self, "Worker error", m))
        self.wk.scan_finished.connect(self._scan_finished)
        self.wk.start()
        self.wk.trigger_start_signal.emit()

    # --- span handling ---
    def apply_span(self):
        try:
            f0 = float(self.le0.text())*1e6
            f1 = float(self.le1.text())*1e6
            n  = int(self.leN.text())
            if not (MIN_FREQ <= f0 < f1 <= MAX_FREQ and n >= 2):
                raise ValueError
        except ValueError:
            print("Span input error")
            return
        self.wk.stop(); self.wk.wait()
        self.wk.set_span(f0, f1, n)
        self.wk.stop_flag = False
        self.wk.start()
        self.ax21.set_xlim(f0/1e9, f1/1e9)
        self.canvas.draw()

    def _scan_finished(self):
        print("Scan finished")
        self.point_index = 0
        self.first_plot = False
        if not self.checkbox_single.isChecked():
            self.wk.trigger_start_signal.emit()

    def _reset(self):
        self.point_index = 0
        self.x21.clear()
        self.y21r.clear()

        self.x21_freq.clear()
        self.y21_value.clear()

        for ln in (self.l21, self.d21):
            ln.set_data([], [])
        self._clear_markers()
        self.canvas.draw()

    def _update_plot(self, f, s21):
        fGHz = f/1e9
        self.freq_label.setText(f"Freq:{fGHz*1000}MHz")
        if self.first_plot:
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

        try:
            y_data[self.point_index] = s21

        except IndexError:
            y_data.append(s21)
            x_data.append(fGHz)

        s21_plot_line.set_data(x_data, smooth_trace(np.array(y_data), SMOOTH_WIN21))
        s21_plot_dot.set_data(x_data, y_data)

        self.ax21.relim(); 
        self.ax21.autoscale_view(scalex=False)

        self.canvas.draw_idle()
        self.point_index  += 1

    # --- visibility toggles ---
    def _vis_toggle(self):
        showS = self.cbSmooth.isChecked()
        showR = self.cbRaw.isChecked()
        self.l21.set_visible(showS); 
        self.d21.set_visible(showR); 
        self.s21_line_continuous.set_visible(showS); 
        self.s21_dots_continuous.set_visible(showR); 
        self.canvas.draw_idle()

    # --- marker handling ---
    def _clear_markers(self):
        for m in self.markers: m.remove()
        self.markers.clear()

    def _on_click(self, event):
        # if event.inaxes not in (self.ax21):
        #     return
        ax = event.inaxes
        if event.button == 1:
            # if ax is self.ax21:
            xdata = self.x21
            ydata = self.y21r if self.d21.get_visible() else list(self.l21.get_ydata())
            if not xdata:
                return
            idx = int(np.argmin(np.abs(np.array(xdata) - event.xdata)))
            x = xdata[idx]; y = ydata[idx]
            if np.isnan(y):
                return
            mrk = ax.plot(x, y, 'kx', ms=8, mew=2)[0]
            txt = ax.annotate(f"{y:.2f} dB\n{x:.3f} GHz",
                              (x, y),
                              textcoords="offset points",
                              xytext=(5, 5),
                              fontsize=8,
                              color='k',
                              bbox=dict(boxstyle="round,pad=0.2", fc='w', alpha=0.7))
            self.markers.extend([mrk, txt])
            self.canvas.draw_idle()
        elif event.button == 3:
            self._clear_markers()
            self.canvas.draw_idle()

    # --- calibration helpers (unchanged) ---
    def _do_cal(self, msg):
        freqs = np.linspace(MIN_FREQ, MAX_FREQ, CAL_POINTS)
        out = {'freqs': [], 'linear': [], 'db': []}
        dlg = QProgressDialog(msg, "Cancel", 0, len(freqs), self)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal); dlg.show()
        for i, f in enumerate(freqs):
            if dlg.wasCanceled():
                return None
            dlg.setValue(i); QApplication.processEvents()
            NUM_R = 4 if f < 1e9 else 1
            sdr.tx_lo = sdr.rx_lo = int(f); time.sleep(DWELL)
            for _ in range(CLR_READS):
                sdr.rx()
            acc = np.zeros(SAMPLE_BUFFER_SIZE*NUM_R, np.complex64)
            for j in range(NUM_R):
                r = sdr.rx()
                acc[j*SAMPLE_BUFFER_SIZE:(j+1)*SAMPLE_BUFFER_SIZE] = (r/2**12)
            A = lockin(acc)
            out['freqs'].append(f)
            out['linear'].append(A)
            out['db'].append(to_dB(A))
        dlg.close()
        return {k: np.array(v) for k, v in out.items()}

    # --- S21 helpers ---
    def cal_s21(self):
        self.wk.stop(); self.wk.wait()
        data = self._do_cal("Calibrating S21…")
        if data is not None:
            np.savez("cal_s21.npz", **data)
            self.load_s21()
        self.wk.stop_flag = False
        self.wk.start()

    def load_s21(self):
        d = np.load("cal_s21.npz")
        self.wk.load_cal21({'freqs': d['freqs'], 'db': d['db']})
        print("S21 calibration loaded")

    # --- cleanup ---
    def closeEvent(self, e):
        self.wk.stop(); self.wk.wait()
        sdr.tx_destroy_buffer(); sdr.rx_destroy_buffer()
        e.accept()

# ───────────── main ─────────────
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
