import numpy as np
from pylab import *
from scipy.signal import iirfilter, freqz, freqs, lfilter
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
from scipy.signal import kaiserord, firwin, lfilter, fftconvolve, oaconvolve
# https://schaumont.dyn.wpi.edu/ece4703b20/lecture5.html
sample_rate = 8e6
nyq_rate = sample_rate / 2.0

#Filter ribble 70 means also that attenuation in stop band is 70dB
FILT_RIPPLE_DB, FILT_CUTOFF_HZ, FILT_TRANS_WIDTH_HZ = 70, 500, 1000

#Calculate taps length and beta parameter for filter requirements
# https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
# https://tomroelandts.com/articles/how-to-create-a-configurable-filter-using-a-kaiser-window
N, beta = kaiserord(FILT_RIPPLE_DB, FILT_TRANS_WIDTH_HZ/nyq_rate)
print(f"N: {N}")
b_fir   = firwin(N, FILT_CUTOFF_HZ/nyq_rate, window=('kaiser', beta))

print(b_fir)
#Plot frequency and phase response
def mfreqz(b,a=1 ):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html
    w,h = freqz(b,a, fs= sample_rate)
    h_dB = 20 * log10 (abs(h))
    subplot(211)
    plot(w,h_dB)
    ylim(-150, 5)
    ylabel("Magnitude (db)")
    xlabel("Normalized Frequency")
    title("Frequency response")
    subplot(212)
    h_Phase = unwrap(arctan2(imag(h),real(h)))
    plot(w,h_Phase)
    ylabel("Phase (radians)")
    xlabel("Normalized Frequency")
    title("Phase response")
    subplots_adjust(hspace=0.5)
mfreqz(b_fir)

# show()

#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------
base_freq = 100_000
period_base_freq = 1.0/base_freq
sampling_period = 1.0/sample_rate
nsamples = 500* period_base_freq/sampling_period
print(f"Filering example samples count: {nsamples}")

if nsamples < N:
    print("Warning input signal samples count too low")
t = np.arange(nsamples) / sample_rate

x = (2^12)*np.sin(2*np.pi*1_000_000*t)

# Use lfilter to filter x with the FIR filter.
# filtered_x = lfilter(b = b,a = a,x = x)

filtered_x = fftconvolve(x, b_fir, mode='same')

# # The phase delay of the filtered signal.
delay = 0.5 * (N-1) / sample_rate

figure(3)
# Plot the original signal.
plot(t, x)
# Plot the filtered signal, shifted to compensate for the phase delay.
plot(t, filtered_x, 'r-')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
# plot(t[N-1:]-delay, filtered_x[N-1:], 'g', linewidth=4)

base_freq = 100
period_base_freq = 1.0/base_freq
sampling_period = 1.0/sample_rate
nsamples = 100* period_base_freq/sampling_period
print(f"Filering example samples count: {nsamples}")

if nsamples < 2* N:
    print("Warning input signal samples count too low")
t = np.arange(nsamples) / sample_rate

x = (2^12)*np.sin(2*np.pi*base_freq*t)

# Use lfilter to filter x with the FIR filter.
# filtered_x = lfilter(b = b,a = a,x = x)

filtered_x = fftconvolve(x, b_fir, mode='same')

# # The phase delay of the filtered signal.
delay = 0.5 * (N-1) / sample_rate

figure(4)
# Plot the original signal.
plot(t, x)
# Plot the filtered signal, shifted to compensate for the phase delay.
plot(t, filtered_x, 'r-')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
# plot(t[N-1:]-delay, filtered_x[N-1:], 'g', linewidth=4)

show()