#!/usr/bin/env python3
import sys, os, time, traceback
import signal
import adi




# ───────────── hardware initialisation ─────────────

exit = False

def handle_int_signal(_, _1):
    global exit
    exit = True

def create_device():
    SDR_URI = "ip:pluto.local"

    while exit is False:
        try:
            sdr2 = adi.Pluto(uri=SDR_URI)
            sdr2.sample_rate             = int(5000000)
            sdr2.rx_rf_bandwidth         = int(5000000)
            sdr2.tx_rf_bandwidth         = int(5000000)
            sdr2.rx_buffer_size          = 1024
            sdr2.gain_control_mode_chan0 = "manual"
            RX_HARDWARE_GAIN = 0
            sdr2.rx_hardwaregain_chan0   = RX_HARDWARE_GAIN
            sdr2.tx_hardwaregain_chan0   = -89
            sdr2._set_iio_attr("out", "voltage_filter_fir_en", False, 0)
            sdr2._set_iio_dev_attr_str("xo_correction", 40000000-380)
            print(f"XO correcton: {sdr2._get_iio_dev_attr("xo_correction")}")
            print(sdr2)
            return sdr2
        except Exception as e:
            print(e)
            print("SDR connection failed trying again")
    
    

signal.signal(signal.SIGINT, handle_int_signal)

sdr = create_device()
print("SDR device created")

while exit is False:
    if not sdr:
        print("Trying to create device again")
        sdr = create_device()
    try:
        print(sdr)
    except Exception as e:
        print(e)
        sdr = None
        print("Device lost ")
    time.sleep(2)