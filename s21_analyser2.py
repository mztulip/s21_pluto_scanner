#!/usr/bin/env python3
import sys, os, time, traceback
import numpy as np
import pandas as pd
import matplotlib
import signal
import logging
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
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

log = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.INFO)

# ───────────── configuration constants ─────────────
FILTER_MODE, FFT_METHOD = 'fft', 'fftconvolve'
FILT_RIPPLE_DB = 70
FILT_CUTOFF_HZ = 12_000
FILT_TRANS_WIDTH_HZ = 100
DEFAULT_STEPS, CAL_POINTS = 500, 500
SMOOTH_WIN21, SMOOTH_WIN11 = 7, 5
AD_SAMPLING_FREQUENCY = 8e6
TX_TONE_FREQ = 100_000
RF_FILTER_BANDWIDTH=1e6
DWELL, CLR_READS, EPS = 0.2, 5, 1e-15
MIN_FREQ, MAX_FREQ, SWEEP_INIT_DELAY = 0.3e9, 6e9, 2.0

TX_BUFFER_SIZE = 1_000

RX_HARDWARE_GAIN = 30
TX_ATTENUATOR_GAIN = -3

# ───────────── FIR filter for lock-in ─────────────
nyq     = AD_SAMPLING_FREQUENCY / 2
N, beta = kaiserord(FILT_RIPPLE_DB, FILT_TRANS_WIDTH_HZ/nyq)
b_fir   = firwin(N, FILT_CUTOFF_HZ/nyq, window=('kaiser', beta))

fft_size = 4096*8
fft_magnitude_db = None
rx_buffer_size = fft_size+N
print(f"Fir taps: {N}")
print(f"Samples buffer size: {rx_buffer_size}")

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


class RxFailed(Exception):
    pass


class TxFailed(Exception):
    pass

class Point():
    def __init__(self, dev_tx, dev_rx):
        self.dev_tx = dev_tx
        self.dev_rx = dev_rx

    def _safe_rx(self):
        try:
            return self.dev_rx.rx()
        except Exception as e:
            log.warning(f"SDR RX failed reading rx: {e}")
            raise RxFailed

    def get(self, frequency_hz, delay = 0):
        try:
            self.dev_tx.tx_lo = int(frequency_hz)
        except Exception as e:
            log.warning(f"TX setting LO freq failed: {e}")
            raise TxFailed
        try:
            self.dev_rx.rx_lo = int(frequency_hz)
            #Necessary because sometimes without control it jumps to 73dB
            self.dev_rx.rx_hardwaregain_chan0   = RX_HARDWARE_GAIN
        except Exception as e:
            log.warning(f"RX setting LO freq, gain failed: {e}")
            raise RxFailed

        time.sleep(DWELL + delay)

        for _ in range(CLR_READS):
            data = self._safe_rx()
            print(f"Clear data: {data}")

        iq_buffer = self._safe_rx()
        # if type(iq_buffer) is list:
        #     iq_buffer = iq_buffer[0]
        # print(iq_buffer)
        iq_buffer /= (2**12)
        log.info(f"Freqpoint: {frequency_hz}")

        iq_filtered = apply_filter(iq_buffer)
        # iq_filtered = iq_buffer
        iq_filtered = iq_filtered[N:]  # discard FIR transient

        global fft_magnitude_db
        fft_iq_buffer = iq_filtered * np.hamming(fft_size)

        fft_bins = np.fft.fftshift(np.fft.fft(fft_iq_buffer))/(fft_size)
        magnitude = np.abs(fft_bins)
        fft_magnitude_db = 20*np.log10(magnitude)
        freq_peak_index = np.argmax(fft_magnitude_db)
        s21 = fft_magnitude_db[freq_peak_index]
        log.info(f"FFT Peak index: {freq_peak_index} Value:{s21}")  

        return s21


# ───────────── worker thread ─────────────
class SweepThread(QThread):
    update  = pyqtSignal(float, float)
    scan_finished = pyqtSignal()
    trigger_start_signal = pyqtSignal(bool)
    error   = pyqtSignal(str)
    pause_signal = pyqtSignal(bool)
    update_devices_signal = pyqtSignal(object, object)
    get_devs_request_signal = pyqtSignal()
    get_devs_response_signal = pyqtSignal(object, object)


    def __init__(self, f_start, f_end, steps):
        super().__init__()
        self.dev_tx = None
        self.dev_rx = None
        self.f0 = f_start
        self.f1 = f_end
        self.n = steps
        self.stop_flag = False
        self.cal21     = None
        self.trigger_start = False
        self.trigger_start_signal.connect(self._trigger_start)
        self.pause_signal.connect(self._pause_signal_handle)
        self.scan_paused = False
        self.calibrate = False
        self.next_point = False
        self.prev_point = False


    def slot_next_point(self):
        log.info("Next frequency point")
        self.next_point = True


    def slot_prev_point(self):
        log.info("Previous frequency point")
        self.prev_point = True


    def _pause_signal_handle(self, paused: bool):
        self.scan_paused = paused


    def _trigger_start(self, calibrate: bool):
        log.info("Start trigger emmited")
        self.trigger_start = True
        self.calibrate = calibrate


    def stop(self):
        self.stop_flag = True


    def load_cal21(self):
        if os.path.exists('cal_s21.npz'):
            cal_data = np.load("cal_s21.npz")
            log.info(f"Loaded cals: {cal_data}")
            self.cal21 = {'freqs': cal_data['freqs'], 'db': cal_data['db']}
            log.info("S21 calibration loaded")
        else:
            log.info("S21 calibration not present")
            

    def check_create_sdr_devices(self):
        log.info("Devices check")
    
        if not self.dev_tx:
            print("No TX SDR device creating new")
            self.dev_tx = self.create_tx_sdr_device()
      
        if not self.dev_rx:
            print("No RX SDR device creating new")
            self.dev_rx = self.create_rx_sdr_device()


    def create_tx_sdr_device(self):
        while self.stop_flag is False:
            try:
                 # sdr_tx_device = adi.Pluto("ip:192.168.2.137")
                sdr_tx_device = adi.ad9361("ip:192.168.2.137")
                sdr_tx_device.tx_enabled_channels     = [1]
                sdr_tx_device.tx_rf_bandwidth         = int(RF_FILTER_BANDWIDTH)
                sdr_tx_device.tx_buffer_size          = TX_BUFFER_SIZE
                sdr_tx_device.tx_cyclic_buffer        = True
                sdr_tx_device.tx_hardwaregain_chan0   = TX_ATTENUATOR_GAIN

                #this is not  used in sdr_tx_device but I think should be configured to be in known state
                sdr_tx_device.sample_rate             = int(AD_SAMPLING_FREQUENCY)
                sdr_tx_device.rx_rf_bandwidth         = int(RF_FILTER_BANDWIDTH)
                sdr_tx_device.rx_buffer_size          = rx_buffer_size
                sdr_tx_device.gain_control_mode_chan0 = "manual"
                sdr_tx_device.rx_hardwaregain_chan0   = 0
                sdr_tx_device._set_iio_attr("out", "voltage_filter_fir_en", False, 0)
                _t = np.arange(TX_BUFFER_SIZE) / AD_SAMPLING_FREQUENCY
                tx_samples = (0.5*np.exp(2j * np.pi * TX_TONE_FREQ * _t)).astype(np.complex64)
                tx_samples *= 2**14 
                sdr_tx_device.tx(tx_samples)
                print("TX device created and configured")
                return sdr_tx_device
            
            except Exception as e:
                print(e)
                print("TX SDR connection failed trying again")


    def create_rx_sdr_device(self):
        while self.stop_flag is False:
            try:
                SDR_URI = "ip:pluto.local"
                # SDR_URI = "ip:192.168.2.121"
                sdr_rx_device = adi.Pluto(uri=SDR_URI)
                sdr_rx_device.sample_rate             = int(AD_SAMPLING_FREQUENCY)
                sdr_rx_device.rx_rf_bandwidth         = int(RF_FILTER_BANDWIDTH)
                sdr_rx_device.tx_rf_bandwidth         = int(RF_FILTER_BANDWIDTH)
                sdr_rx_device.rx_buffer_size          = rx_buffer_size
                sdr_rx_device.gain_control_mode_chan0 = "manual"
                sdr_rx_device.rx_hardwaregain_chan0   = RX_HARDWARE_GAIN
                sdr_rx_device.tx_hardwaregain_chan0   = -89
                sdr_rx_device._set_iio_attr("out", "voltage_filter_fir_en", False, 0)
                sdr_rx_device._set_iio_dev_attr_str("xo_correction", 40000000-290)
                print(f"XO correcton: {sdr_rx_device._get_iio_dev_attr("xo_correction")}")
                print("RX device created and configured")
                return sdr_rx_device

            except Exception as e:
                print(e)
                print("RX SDR connection failed trying again")


    def check_devs(self):
        point = None
        wait_counter = 0
        while True:
            if self.stop_flag:
                return
            
            if self.dev_tx and self.dev_rx:
                point = Point(self.dev_tx, self.dev_rx)
                break

            if wait_counter%20 == 0:
                log.warning("Worker: Waiting for sdr devices")
            self.check_create_sdr_devices()
            time.sleep(0.1)
            wait_counter+=1
        return point


    def run(self):
        log.info("Worker run started")
        self.load_cal21()
        self.check_create_sdr_devices()
        time.sleep(SWEEP_INIT_DELAY)

        log.info("Worker while loop begin")
        wait_counter = 0
        while self.trigger_start is False:
            if wait_counter%10 == 0:
                log.info("Waiting for start trigger")
            if self.stop_flag:
                self.trigger_start = False
                return
            time.sleep(0.1)
            wait_counter+=1
        self.trigger_start = False

        if self.calibrate:
            self.cal21 = None
            freqs = np.linspace(MIN_FREQ, MAX_FREQ, CAL_POINTS)
        else:
            freqs = np.linspace(self.f0, self.f1, self.n)
        out = {'freqs': [], 'db': []}
        freq_index = 0
        
        while freq_index < len(freqs):
            i = freq_index
            f = freqs[i]
            if self.stop_flag or self.trigger_start is True:
                log.info("Trigger start true, starting from beginning")
                break

            point = self.check_devs()

            # Delay is necessary when changging from end to begining
            # without it first point is inccorect have lower power
            delay = 0
            if freq_index == 0:
                delay = 1
            try:
                s21 = point.get(f, delay)

            except RxFailed:
                log.warning("Getting point failed, Rx failed")
                self.dev_rx = None
                continue
            except TxFailed:
                log.warning("Getting point failed, Tx failed")
                self.dev_tx = None
                continue

            out['freqs'].append(f)
            out['db'].append(s21)

            if s21 < -60 and self.calibrate:
                print(f"Warning: RX signal is below -60dB something is wrong freq:{f}")

            offset = None
            if self.cal21 is not None:
                offset = np.interp(f, self.cal21['freqs'], self.cal21['db'])
                s21 -= offset
            try:
                rssi = self.dev_rx._get_iio_attr('voltage0','rssi', False)
                log.info(f"S21 calibrated: {s21} calib offset: {offset} RSSI: -{rssi}dB")
            except Exception as e:
                log.warning("Getting RSSI failed")
                self.dev_rx = None
                continue

            self.update.emit(f, s21)

            if self.scan_paused is False:
                freq_index += 1

            if self.next_point:
                self.next_point = False
                freq_index += 1

            if self.prev_point:
                self.prev_point = False
                freq_index -= 1

        if self.dev_tx:
            self.dev_tx.tx_destroy_buffer()
        if self.dev_rx:
            self.dev_rx.rx_destroy_buffer()


        self.scan_finished.emit()

        if self.calibrate:
            log.info("Saving calibration data")
            np.savez("cal_s21.npz", **out)

        log.info("Worker thread exited")


# ───────────── GUI with toggles + markers ─────────────
class VNA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.exit_loops = False
        self.freq_start = MIN_FREQ
        self.freq_stop = MAX_FREQ
        self.steps = DEFAULT_STEPS

        self.setWindowTitle("2 pluto device scalar analyser")
        self.cal_dlg = None
        self.fft_marker = []
        self._build_ui()
        self._init_plot()
        self._spawn_worker()
        self.button_next.clicked.connect(self.wk.slot_next_point)
        self.button_prev.clicked.connect(self.wk.slot_prev_point)
        self.wk.trigger_start_signal.emit(False)
        signal.signal(signal.SIGINT, self.sig_int)

    # --- UI bar + checkboxes ---
    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        box1 = QVBoxLayout(cw)

        top = QFrame()
        top.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)

        h_box1 = QHBoxLayout()
        h_box2 = QHBoxLayout()

        box1.addLayout(h_box1)
        box1.addLayout(h_box2)

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

        self.fig, (self.axes_s21, self.axes_fft) = plt.subplots(2,1, figsize=(12,9))
        # self.axes_fft.axis('off')
        self.canvas = FigureCanvas(self.fig)
        box1.addWidget(self.canvas)
        box1.addWidget(NavigationToolbar2QT(self.canvas, self))

        button_pause = QPushButton("Pause")
        button_pause.clicked.connect(self._pause_emit)
        h_box2.addWidget(button_pause)

        button_continue = QPushButton("Continue")
        button_continue.clicked.connect(self._continue_emit)
        h_box2.addWidget(button_continue)

        self.button_next = QPushButton("Next")
        h_box2.addWidget(self.button_next)

        self.button_prev = QPushButton("Prev")
        h_box2.addWidget(self.button_prev)

        label_rx_att = QLabel('RX att: 30dB')
        h_box2.addWidget(label_rx_att)

        label_tx_att = QLabel('TX att: 10dB')
        h_box2.addWidget(label_tx_att)

        label_rx_gain = QLabel(f'RX gain: {RX_HARDWARE_GAIN}dB')
        h_box2.addWidget(label_rx_gain)

        label_tx_ad_att = QLabel(f'TX att ad: {TX_ATTENUATOR_GAIN}dB')
        h_box2.addWidget(label_tx_ad_att)
        

    def _pause_emit(self):
        self.wk.pause_signal.emit(True)

    def _continue_emit(self):
        self.wk.pause_signal.emit(False)

    def _start_from_beginning(self):
        self.wk.stop()
        print("Waiting for worker to stop")
        self.wk.wait()
        print("Worker stopped")
        self.wk.scan_finished.disconnect()
        self.wk.update.disconnect()
        self._reset_plot()
        self.first_plot = True
        print("Start from beginning")
        self._spawn_worker()
        self.wk.trigger_start_signal.emit(False)

    # --- traces + markers storage + titles ---
    def _init_plot(self):
        self.first_plot = True
        self.first_data_x_s21, self.first_data_y_s21 = [], []
        self.data_x21_freq, self.data_y21_value = [], []

        self.first_line_21, = self.axes_s21.plot([], [], 'b-', lw=1.2, label='S21 smoothed')
        self.first_dots_21, = self.axes_s21.plot([], [], 'ro', ms=4,   label='S21 raw')

        self.s21_line_continuous, = self.axes_s21.plot([], [], 'y-', lw=1.2, label='S21 smoothed2')
        self.s21_dots_continuous, = self.axes_s21.plot([], [], 'go', ms=4,   label='S21 raw2')

        self.axes_s21.set_title("S21 (dB)")
        self.axes_s21.set_xlim(self.freq_start/1e9, self.freq_stop/1e9)
        self.axes_s21.set_ylabel("dB")
        self.axes_s21.grid(True)

        self.freqs = np.arange(-AD_SAMPLING_FREQUENCY/2, AD_SAMPLING_FREQUENCY/2, AD_SAMPLING_FREQUENCY / fft_size)
        self.fft_line, = self.axes_fft.plot([], [], 'b-', lw=1.2, label='fft plot')
        self.axes_fft.set_title("frequency")
        self.axes_fft.set_xlim(-AD_SAMPLING_FREQUENCY/200, AD_SAMPLING_FREQUENCY/200)
        # self.axes_fft.set_ylim(-120,-50)
        self.axes_fft.set_ylabel("dB")
        self.axes_fft.grid(True)


        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.markers = []
        self._vis_toggle()
        self.point_index = 0

    # --- worker startup ---
    def _spawn_worker(self):
        self.wk = SweepThread(self.freq_start, self.freq_stop, self.steps)
        self.wk.update.connect(self._update_plot)
        self.wk.error.connect(lambda m: QMessageBox.critical(self, "Worker error", m))
        self.wk.scan_finished.connect(self._scan_finished)
        self.wk.start()

    # --- span handling ---
    def apply_span(self):
        try:
            self.freq_start = float(self.le0.text())*1e6
            self.freq_stop = float(self.le1.text())*1e6
            self.steps  = int(self.leN.text())
            if not (MIN_FREQ <= self.freq_start < self.freq_stop <= MAX_FREQ and self.steps >= 2):
                raise ValueError
        except ValueError:
            print("Span input error")
            return

        self.wk.stop(); 
        self.wk.wait()
        self.wk.scan_finished.disconnect()
        self.wk.update.disconnect()
        self._reset_plot()
        self._spawn_worker()
        self.wk.trigger_start_signal.emit(False)
        # self.wk.stop_flag = False
        # self.wk.start()
        self.axes_s21.set_xlim(self.freq_start/1e9, self.freq_stop/1e9)
        self.canvas.draw()

    def _scan_finished(self):
        print("Scan finished")
        self.point_index = 0
        self.first_plot = False
        print("first_plot set to false")
        self.wk.scan_finished.disconnect()
        self.wk.update.disconnect()
        if self.cal_dlg:
            self.cal_dlg.close()
            self.cal_dlg = None

        if not self.checkbox_single.isChecked():
            self._spawn_worker()
            self.wk.trigger_start_signal.emit(False)

    def _reset_plot(self):
        self.point_index = 0
        self.first_data_x_s21.clear()
        self.first_data_y_s21.clear()

        self.data_x21_freq.clear()
        self.data_y21_value.clear()

        for ln in (self.first_line_21, self.first_dots_21, self.s21_line_continuous, self.s21_dots_continuous):
            ln.set_data([], [])
        self._clear_markers()
        self.canvas.draw()

    def _update_plot(self, f_vco_hz, s21):
        fGHz = f_vco_hz/1e9
        self.freq_label.setText(f"Freq:{fGHz*1000}MHz")
        if self.first_plot:
            x_data = self.first_data_x_s21
            y_data = self.first_data_y_s21
            s21_plot_line = self.first_line_21
            s21_plot_dot = self.first_dots_21
            print("First plot")
        else:
            x_data = self.data_x21_freq
            y_data = self.data_y21_value
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

        self.axes_s21.relim(); 
        self.axes_s21.autoscale_view(scalex=False)
     
        global fft_magnitude_db
        freq_peak_index = np.argmax(fft_magnitude_db)
        freq_peak = self.freqs[freq_peak_index]
        s21_peak = fft_magnitude_db[freq_peak_index]
        self.fft_line.set_data(self.freqs, fft_magnitude_db)
        
        for m in self.fft_marker:
            print(m)
            m.remove()
        self.fft_marker.clear()
     
        mark = self.axes_fft.plot(freq_peak, s21_peak, 'kx', ms=8, mew=2)[0]
        txt_mark = self.axes_fft.annotate(f"{s21_peak:.2f} dB\n{freq_peak:.3f} Hz",
                              (freq_peak+5000, s21_peak-10),
                              textcoords="offset points",
                              xytext=(5, 5),
                              fontsize=8,
                              color='k',
                              bbox=dict(boxstyle="round,pad=0.2", fc='w', alpha=0.7))
        self.fft_marker.extend([mark, txt_mark])


        self.axes_fft.relim(); 
        self.axes_fft.autoscale_view(scalex=False)

        self.canvas.draw_idle()

        if self.cal_dlg:
            self.cal_dlg.setValue(self.point_index)
            QApplication.processEvents()

        self.point_index  += 1

    # --- visibility toggles ---
    def _vis_toggle(self):
        showS = self.cbSmooth.isChecked()
        showR = self.cbRaw.isChecked()
        self.first_line_21.set_visible(showS); 
        self.first_dots_21.set_visible(showR); 
        self.s21_line_continuous.set_visible(showS); 
        self.s21_dots_continuous.set_visible(showR); 
        self.canvas.draw_idle()

    # --- marker handling ---
    def _clear_markers(self):
        for m in self.markers: m.remove()
        self.markers.clear()

    def _on_click(self, event):
        ax = event.inaxes
        if event.button == 1:
            xdata = self.first_data_x_s21
            ydata = self.first_data_y_s21 if self.first_dots_21.get_visible() else list(self.first_line_21.get_ydata())
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


    def cancel_calibration(self):
        log.info("Cancel calibration pressed. Starting normal thread")
        self.cal_dlg.close()
        self.cal_dlg = None
        self.wk.stop()
        self.wk.wait()
        self.wk.scan_finished.disconnect()
        self.wk.update.disconnect()
        self._reset_plot()
        self._spawn_worker()
        self.wk.trigger_start_signal.emit(False)

    def cal_s21(self):
        log.info("Calibration started")
        self.wk.stop()
        self.wk.wait()
        self.wk.scan_finished.disconnect()
        self.wk.update.disconnect()
        self._reset_plot()
        self._spawn_worker()
        self.wk.trigger_start_signal.emit(True)
      
        self.cal_dlg = QProgressDialog("Calibrating S21…", "Cancel", 0, CAL_POINTS, self)
        self.cal_dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.cal_dlg.show()

        cancelButton = self.cal_dlg.findChild(QPushButton)
        cancelButton.clicked.connect(self.cancel_calibration)


    def load_s21(self):
        print("Load function not implemented")

    def sig_int(self, signum, frame):
        print("Ctrl+c pressed SIGINT received, exitting")
        self._close()
        self.close()

    def _close(self):
        self.exit_loops = True
        self.wk.stop()
        self.wk.wait()

    # --- cleanup ---
    def closeEvent(self, e):
        self._close()
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
