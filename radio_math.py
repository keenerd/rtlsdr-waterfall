#! /usr/bin/env python2

from itertools import *
import math, numpy

tau = 2 * math.pi

class Translate(object):
    "set up once, consumes an I/Q stream, returns an I/Q stream"
    def __init__(self, num, den):
        angles = [(a*tau*num/den) % tau for a in range(den)]
        fir = [complex(math.cos(a), math.sin(a)) for a in angles]
        self.looping_fir = cycle(fir)
    def __call__(self, stream):
        return numpy.array([s1*s2 for s1,s2 in izip(self.looping_fir, stream)])

class Downsample(object):
    "set up once, consumes an I/Q stream, returns an I/Q stream"
    # aka lowpass
    def __init__(self, scale):
        self.scale = scale
        self.offset = 0
        self.window = numpy.hanning(scale * 2)
        self.window = self.window / sum(self.window)
    def __call__(self, stream):
        prev_off = self.offset
        self.offset = self.scale - ((len(stream) + self.offset) % self.scale)
        # bad edges, does 60x more math than needed
        stream2 = numpy.convolve(stream, self.window)
        return stream2[prev_off::self.scale]

class DownsampleFloat(object):
    # poor quality, but good temporal accuracy
    # uses triangle window
    def __init__(self, scale):
        self.scale = scale
        self.offset = 0
    def __call__(self, stream):
        # bad edges
        # should be using more numpy magic
        scale = self.scale
        stream2 = []
        for x in numpy.arange(self.offset, len(stream), scale):
            frac = x % 1.0
            window = numpy.concatenate((numpy.arange(1-frac, scale),
                     numpy.arange(int(scale)+frac-1, 0, -1)))
            window = window / sum(window)
            start = x - len(window)//2
            start = max(start, 0)
            start = min(start, len(stream)-len(window))
            c = sum(stream[start : start+len(window)] * window)
            stream2.append(c)
        return numpy.array(stream2)

class Upsample(object):
    # use minimal power interpolation?
    def __init__(self, scale):
        self.scale = scale
        self.offset = 0
    def __call__(self, stream):
        xp = range(len(stream))
        x2 = numpy.arange(self.offset, len(stream), 1.0/self.scale)
        self.offset = (len(stream) + self.offset) % self.scale
        reals = numpy.interp(x2, xp, stream.real)
        imags = numpy.interp(x2, xp, stream.imag)
        return numpy.array([complex(*ri) for ri in zip(reals, imags)])

class Bandpass(object):
    "set up once, consumes an I/Q stream, returns an I/Q stream"
    def __init__(self, center_fc, center_bw, pass_fc, pass_bw):
        # some errors from dropping the fractional parts of the ratio
        # either optimize tuning, scale up, or use floating point
        ratio = (center_fc - pass_fc) / center_bw
        self.translate = Translate(ratio * 1024, 1024)
        self.downsample = Downsample(int(center_bw/pass_bw))
    def __call__(self, stream):
        return self.downsample(self.translate(stream))

# check license on matplotlib code
# chop out as much slow junk as possible

# http://mail.scipy.org/pipermail/numpy-discussion/2003-January/014298.html
# http://cleaver.cnx.rice.edu/eggs_directory/obspy.signal/obspy.signal/obspy/signal/freqattributes.py
# /usr/lib/python2.7/site-packages/matplotlib/mlab.py

def detrend_none(x):
    "Return x: no detrending"
    return x

def window_hanning(x):
    "return x times the hanning window of len(x)"
    return numpy.hanning(len(x))*x

def psd(x, NFFT=256, Fs=2, Fc=0, detrend=detrend_none, window=window_hanning,
        noverlap=0, pad_to=None, sides='default', scale_by_freq=True):
    Pxx,freqs = csd(x, x, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
        scale_by_freq)
    return Pxx.real, freqs + Fc

def csd(x, y, NFFT=256, Fs=2, detrend=detrend_none, window=window_hanning,
        noverlap=0, pad_to=None, sides='default', scale_by_freq=True):
    Pxy, freqs, t = _spectral_helper(x, y, NFFT, Fs, detrend, window,
        noverlap, pad_to, sides, scale_by_freq)

    if len(Pxy.shape) == 2 and Pxy.shape[1]>1:
        Pxy = Pxy.mean(axis=1)
    return Pxy, freqs


#This is a helper function that implements the commonality between the
#psd, csd, and spectrogram.  It is *NOT* meant to be used outside of mlab
def _spectral_helper(x, y, NFFT=256, Fs=2, detrend=detrend_none,
        window=window_hanning, noverlap=0, pad_to=None, sides='default',
        scale_by_freq=True):
    #The checks for if y is x are so that we can use the same function to
    #implement the core of psd(), csd(), and spectrogram() without doing
    #extra calculations.  We return the unaveraged Pxy, freqs, and t.
    same_data = y is x

    #Make sure we're dealing with a numpy array. If y and x were the same
    #object to start with, keep them that way
    x = numpy.asarray(x)
    if not same_data:
        y = numpy.asarray(y)
    else:
        y = x

    # zero pad x and y up to NFFT if they are shorter than NFFT
    if len(x)<NFFT:
        n = len(x)
        x = numpy.resize(x, (NFFT,))
        x[n:] = 0

    if not same_data and len(y)<NFFT:
        n = len(y)
        y = numpy.resize(y, (NFFT,))
        y[n:] = 0

    if pad_to is None:
        pad_to = NFFT

    # For real x, ignore the negative frequencies unless told otherwise
    if (sides == 'default' and numpy.iscomplexobj(x)) or sides == 'twosided':
        numFreqs = pad_to
        scaling_factor = 1.
    elif sides in ('default', 'onesided'):
        numFreqs = pad_to//2 + 1
        scaling_factor = 2.
    else:
        raise ValueError("sides must be one of: 'default', 'onesided', or "
            "'twosided'")

    #if cbook.iterable(window):
    if type(window) != type(lambda:0):
        assert(len(window) == NFFT)
        windowVals = window
    else:
        windowVals = window(numpy.ones((NFFT,), x.dtype))

    step = NFFT - noverlap
    ind = numpy.arange(0, len(x) - NFFT + 1, step)
    n = len(ind)
    Pxy = numpy.zeros((numFreqs, n), numpy.complex_)

    # do the ffts of the slices
    for i in range(n):
        thisX = x[ind[i]:ind[i]+NFFT]
        thisX = windowVals * detrend(thisX)
        fx = numpy.fft.fft(thisX, n=pad_to)

        if same_data:
            fy = fx
        else:
            thisY = y[ind[i]:ind[i]+NFFT]
            thisY = windowVals * detrend(thisY)
            fy = numpy.fft.fft(thisY, n=pad_to)
        Pxy[:,i] = numpy.conjugate(fx[:numFreqs]) * fy[:numFreqs]

    # Scale the spectrum by the norm of the window to compensate for
    # windowing loss; see Bendat & Piersol Sec 11.5.2.
    Pxy /= (numpy.abs(windowVals)**2).sum()

    # Also include scaling factors for one-sided densities and dividing by the
    # sampling frequency, if desired. Scale everything, except the DC component
    # and the NFFT/2 component:
    Pxy[1:-1] *= scaling_factor

    # MATLAB divides by the sampling frequency so that density function
    # has units of dB/Hz and can be integrated by the plotted frequency
    # values. Perform the same scaling here.
    if scale_by_freq:
        Pxy /= Fs

    t = 1./Fs * (ind + NFFT / 2.)
    freqs = float(Fs) / pad_to * numpy.arange(numFreqs)

    if (numpy.iscomplexobj(x) and sides == 'default') or sides == 'twosided':
        # center the frequency range at zero
        freqs = numpy.concatenate((freqs[numFreqs//2:] - Fs, freqs[:numFreqs//2]))
        Pxy = numpy.concatenate((Pxy[numFreqs//2:, :], Pxy[:numFreqs//2, :]), 0)

    return Pxy, freqs, t


