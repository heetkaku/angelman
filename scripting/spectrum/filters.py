import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.style.use('seaborn-whitegrid')

class Butterworth:
    """Creates an IIR Butterworth filter in second-order sections format.

    Attrs:
        order (int):        filter order
        ftype (str):        the type of filter, one of: 
                            {lowpass, highpass,bandpass, bandstop}
        criticals (seq):    cutoff frequencies of the filter
        fs (int):           sampling frequency of the system
    """

    def __init__(self, order, ftype, criticals, fs):
        """Initialize this Butterworth."""

        self.order = order
        self.ftype = ftype
        self.criticals = criticals
        self.fs = fs
        self.nyq = self.fs / 2
        #compute the numerator & denom. of z-transform polynomials
        self._sos = signal.butter(order, criticals, btype=ftype, 
                                  output='sos', fs=fs)

    def plot(self, **kwargs):
        """Returns a plot of the gain of this filter.

        Args:
            kwargs passed to scipy's sosfreqz
        """

        fig, ax = plt.subplots()
        f, h = signal.sosfreqz(self._sos, fs=self.fs, **kwargs)
        gain = 20 * np.log10(np.abs(h))
        label = '{}th Order Butterworth'
        ax.plot(f, gain, label=label.format(self.order))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.legend()
        plt.show()

    def apply(self, data, axis=-1, **kwargs):
        """Apply this filter both forwards and backwards.
        
        Args:
            data (ndarray):         array of samples to filter
            axis (int):             data axis to apply filter
            kwargs:                 passed to scipy sosfiltfilt
        """
       
        return signal.sosfiltfilt(self._sos, data, axis=axis, **kwargs)

if __name__ == '__main__':
    import time

    butter = Butterworth(order=8, ftype='lowpass', criticals=400, fs=5000)
    butter.plot()

    butter2 = Butterworth(order=8, ftype='lowpass', criticals=50, fs=5000)
    #generate sample data
    time = 1
    fs = butter2.fs
    nsamples = int(time * fs)
    t = np.linspace(0, time, nsamples)

    # make a small 10 Hz riding on top of a larger 100 Hz sinusoidal signal
    x = 0.25 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t)
    fig, ax = plt.subplots(2, 1, figsize=(8,4))
    ax[0].plot(t, x)
    ax[0].set_title('Before Filtering')

    #filter data
    y = butter2.apply(x)
    ax[1].plot(t, y)
    ax[1].set_title('After Filtering')
    plt.show()

