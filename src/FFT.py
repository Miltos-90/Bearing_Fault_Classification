from scipy.fft import fft, fftfreq
from scipy import signal
import numpy as np

''' Minimal settings for an FFT analysis'''
DefaultFFTsettings = {
    'analysisType'      : 'regular', # or envelope
                                     # -------------- Envelope (HP) Filter ------------------
    'HPCutoffFrequency' : 2,         # for regular analysis set to 2 [Hz]. 
                                     # For envelope analysis set to 20 * rotational speed
    'HPFilterOrder'     : 1,
                                     # ----- LP filter (used only in envelope analysis) ------
    'LPCutoffFrequency' : 500,
    'LPFilterOrder'     : 1,
                                     # ---------------------- FFT ------------------------
    'FFTstride'         : 2 ** 16,   # Power of 2 that is still lower than the lenght of the signal
    'FFToverlap'        : 0.75,
    'FFTwindowName'     : 'hann',
    'FFTavgType'        : 'linear',  # For regular analysis set to 'linear'
                                     # For envelope set to 'maxhold'
}


def FFTAnalysis(sig: np.ndarray, 
                samplingFrequency: int, 
                settingsDict: dict = DefaultFFTsettings) -> (np.ndarray, np.ndarray):
    ''' 
    Runs the pipeline for a regular FFT / envelope analysis
    Inputs:
        sig:               Vibration signal
        samplingFrequency: The sampling frequeny of the signal
        settingsDict:      Dictionary with settings
    Outputs:
        fftFreqs:          FFT frequencies
        averagedLevels:    Corresponding levels
    '''
    
    # Check inputs
    _checkInputs(settingsDict, sig.shape[0])
    
    filteredSig = _ButterFilter(sig, samplingFrequency, 
                                settingsDict['HPCutoffFrequency'], 
                                settingsDict['HPFilterOrder'], 
                                filterType = 'HP')
    
    if settingsDict['analysisType'] == 'envelope': # Steps needed only for enveloping
        filteredSig = np.abs(filteredSig)
        filteredSig = _ButterFilter(filteredSig, samplingFrequency,
                                    settingsDict['LPCutoffFrequency'],
                                    settingsDict['LPFilterOrder'],
                                    filterType = 'LP')
    
    sigChunks           = _slidingWindow(filteredSig, settingsDict['FFTstride'], settingsDict['FFToverlap'])
    sigChunks           = _detrendSignal(sigChunks)
    fftFreqs, fftLevels = _FFTwrapper(sigChunks, samplingFrequency, settingsDict['FFTwindowName'])
    averagedLevels      = _FFTaveraging(fftLevels, settingsDict['FFTavgType'])

    return fftFreqs, averagedLevels


def _checkInputs(settingsDict: dict, noSamples: int) -> None:
    '''
    Input check for the settings provided by the user
    Inputs: 
        settingsDict: Dictionary containing the settings provided by the user
        noSamples:    Number of samples in the signal
    '''

    if settingsDict['analysisType'] not in ['envelope', 'regular']:
        raise ValueError('Analysis type can either be envelope or regular')
        
    if settingsDict['FFToverlap'] < 0 or settingsDict['FFToverlap'] >= 1.0:
        raise ValueError('FFToverlap should lie in (0, 1).')
        
    if settingsDict['FFTwindowName'] not in ['flattop', 'hann', 'blackman', 'hamming']:
        raise ValueError('Valid choices fow window are: flattop, hann, blackman, hamming.')   
        
    if settingsDict['FFTavgType'] not in ['maxhold', 'linear']:
        raise ValueError('Averaging type can either be maxhold or linear')
    
    if noSamples <= settingsDict['FFTstride']:
        raise ValueError('Number of samples in the signal should be higher than the stride specified.')
    return 


def _detrendSignal(sig: np.ndarray) -> np.ndarray:
    '''
    Detrends a signal by subtracting the mean value
    Inputs:
        sig: Vibration signals
    Outputs:
        sigChunksZeroMean: Detrended vibration signals
        
    '''
    
    if sig.ndim > 1: # matrix
        rowMeans          = np.mean(sig, axis = 1)
        noCols            = sig.shape[1]
        rowMeansExp       = np.outer(rowMeans, np.ones(noCols))
        sigChunksZeroMean = sig - rowMeansExp

    else: # vector
        sigChunksZeroMean = sig - np.mean(sig)

    return sigChunksZeroMean


def _ButterFilter(sig: np.ndarray, samplingFrequency: int, cutoffFrequency: int, order: int, filterType: str) -> np.ndarray:
    '''
    Butterworth Filter (highpass / lowpass)
    Inputs:
        sig:               Vibration signal
        samplingFrequency: The sampling frequency for sig
        cutoffFrequency:   Cutoff frequency for the filter
        order:             Filter order
        filterType:        Type of filter (highpass or lowpass
    Outputs:
        y:                 Filtered vibration signal
        
    '''
    
    # Computation of the normalised cut-off frequency
    nyq           = 0.5 * samplingFrequency
    normal_cutoff = cutoffFrequency / nyq
    
    if filterType == 'HP': # High-pass filter
        b, a = signal.butter(order, normal_cutoff, btype='high', analog = False)
        y = signal.filtfilt(b, a, sig)
        
    elif filterType == 'LP': # Low-pass filter
        b, a = signal.butter(order, normal_cutoff, btype = 'low', analog = False)
        y = signal.lfilter(b, a, sig)
        
    return y


def _makeWindow(windowName: str, noSamples: int) -> (np.ndarray, float):
    '''
        Make a preconfigured window signal, and an amplitude 
        correction factor for the FFT levels
        Inputs:
            windowName:       Name of the window to be used
            noSamples:        Number of samples of the vibration signal
        Outputs:
            windowSig:        Preconfigured scipy.signal
            windowCorrFactor: Correction factor to be applied to the FFT levels
        Notes:
            Corrections factors are taken from the Siemens PLM blog [1], but they can also
            be evaluated using Equation (9.23) of [2], which gives slightly different results.
            See [3] for a discussion on errata of [2] regarding *power* correction factors 
        References:
            [1] https://community.sw.siemens.com/s/article/window-correction-factors
            [2] Brandt, A., 2011. Noise and vibration analysis: signal analysis and experimental procedures. John Wiley & Sons.
            [3] https://nl.mathworks.com/matlabcentral/answers/372516-calculate-windowing-correction-factor
    '''
    
    if windowName == 'flattop':  
        windowSig  = signal.windows.flattop(M = noSamples)
        corrFactor = 4.18
        
    elif windowName == 'hann':     
        windowSig  = signal.windows.hann(M = noSamples)
        corrFactor = 2.0
        
    elif windowName == 'blackman': 
        windowSig  = signal.windows.blackman(M = noSamples)
        corrFactor = 2.8
        
    elif windowName == 'hamming': 
        windowSig  = signal.windows.hamming(M = noSamples)
        corrFactor = 1.85        
        
    return windowSig, corrFactor


def _FFTaveraging(fftLevels: np.ndarray, avgType:str) -> np.ndarray:
    '''
        Averages FFT levels either with linear averaging or maxhold averaging
        Inputs: 
            fftLevels: The FFT levels
            avgType:   Averaging type to be performed (linear/maxhold)
        Outputs:
            avgLevels: The averaged FFT levels
    '''
    
    if fftLevels.ndim > 1: # 2d matrix
        
        # No. of FFTs to be averaged
        noFFTs    = fftLevels.shape[0]
        
        avgLevels = fftLevels[0, :]

        for fftNo in range(1, noFFTs):
            curLevels = fftLevels[fftNo, :]

            if avgType == 'linear':    avgLevels += curLevels
            elif avgType == 'maxhold': avgLevels = np.maximum(avgLevels, curLevels)

        if avgType == 'linear': 
            avgLevels /= noFFTs
            
    else: # vector -> Return the input
        avgLevels = fftLevels
    
    return avgLevels


def _FFTwrapper(sigChunks: np.ndarray, samplingFrequency: int, windowName: str) -> (np.ndarray, np.ndarray):
    '''
        Returns a number of FFTs based on the chunks of a vibration signal
        Inputs:
            sigChunks:         Vibration signal cut in chunks
            samplingFrequency: Sampling frequency of the vibration signal
            windowName:        Name of the window to be applied
        Outputs:
            fftFreqs:          Frequencies of the FFT
            FFTs:              Corresponding levels
    '''
    if sigChunks.ndim > 1:
        
        noSignals = len(sigChunks) # Number of chunks of the original signal
        for sigNo, sigChunk in enumerate(sigChunks):

            fftFreqs, fftLevels = _FFT(sigChunk, samplingFrequency, windowName)

            if sigNo == 0: # Initialise the empty matrix on the first iteration
                FFTs = np.empty(shape = (noSignals, fftFreqs.shape[0]))

            FFTs[sigNo, :] = fftLevels
            
    else:
         fftFreqs, FFTs = _FFTOneSignal(sigChunks, samplingFrequency, windowName)

    return fftFreqs, FFTs


def _FFT(sig: np.ndarray, samplingFrequency: int, windowName: str) -> (np.ndarray, np.ndarray):
    '''
        Runs an FFT to a given vibration signal with windowing
        Inputs:
            sig:               Vibration signal
            samplingFrequency: Sampling frequency of the vibration signal
            windowName:        Name of the window to be applied
        Outputs:
            fftFreqs:          Frequencies of the FFT
            FFTs:              Corresponding levels
    '''
    
    noSamples = sig.shape[0]
    
    # Get window signal
    windowSig, windowCorrFactor = _makeWindow(windowName, noSamples)
                                             
    # Make windowed signal    
    sigWindow = windowSig * sig
    
    # Compute FFT
    fftLevels = fft(sigWindow)
    fftLevels = 2.0/noSamples * np.abs(fftLevels[0:noSamples//2])
    
    # Compute frequencies
    timeStep = 1 / samplingFrequency
    fftFreqs = fftfreq(noSamples, timeStep)[:noSamples//2]
    
    # Cut-off HF
    fMax      = fftFreqs.max() / 1.24 # Nyquist criteria
    idxtoKeep = fftFreqs < fMax
    fftFreqs  = fftFreqs[idxtoKeep]
    fftLevels = fftLevels[idxtoKeep]
    
    # Amplitude Correction
    fftLevels *= windowCorrFactor
    
    return fftFreqs, fftLevels


def _slidingWindow(sig: np.ndarray, stride: int, overlap: float) -> np.ndarray:
    '''
    Segments a vibration signal into multiple chunks, without or without overlap and a of a given size.
    Inputs:
        sig:     Vibration signal to be segmented
        stride:  Stride of the segments
        overlap: Overlap of the segments
    Outputs:
        out:     Segmented vibration signal
    '''
    
    if overlap == 0:
        out = _slidingWindowWithoutOverlap(sig, stride)
        
    else: # Overlap (0, 1)
        out = _slidingWindowWithOverlap(sig, stride, overlap)
        
    return out


def _slidingWindowWithoutOverlap(sig: np.ndarray, stride: int) -> np.ndarray:
    '''
    Segments a vibration signal into multiple chunks, without overlap and a of a given size.
    It will truncate part of the signal that does not fit the chunks of the specified size and overlap.
    Inputs:
        sig:     Vibration signal to be segmented
        overlap: Overlap of the segments
    Outputs:
        out:     Segmented vibration signal
    '''
    
    # No. samples in the signal
    noSamples = sig.shape[0]
        
    if stride <= noSamples: # Possible to segment the signal
            
        # Segment
        NsecWindowSamples = int(noSamples / stride)
        samplesToKeep     = int(NsecWindowSamples * stride)
        sigTruncated      = sig[0:samplesToKeep]
        sigChunked        = np.split(sigTruncated, NsecWindowSamples)

        # Convert the list of np.split to a 2d np.array
        noRows = len(sigChunked)
        noCols = sigChunked[0].shape[0]
        out    = np.empty(shape = (noRows, noCols)).astype(float)

        for rowNo, elem in enumerate(sigChunked):
            out[rowNo, :] = elem
        
    return out


def _slidingWindowWithOverlap(sig: np.ndarray, stride: int, overlap: float) -> np.ndarray:
    '''
    Segments a vibration signal into multiple chunks, with overlap and a of a given size.
    Inputs:
        sig:     Vibration signal to be segmented
        overlap: Overlap of the segments
    Outputs:
        out:     Segmented vibration signal
    '''
    
    # Get stride and offset [no. samples]
    noSamples = sig.shape[0]
    strideSamples = stride
    offsetSamples = int((1-overlap) * stride)

    if strideSamples <= noSamples:

        # Compute start and end indices of each chunk
        startIdx  = np.arange(start = 0, stop = noSamples, step = offsetSamples).astype(int)
        endIdx    = np.arange(start = strideSamples, stop = noSamples, step = offsetSamples).astype(int)

        # Matrix to hold results
        noSignals = np.min([endIdx.shape[0], startIdx.shape[0]])
        out       = np.empty(shape = (noSignals, strideSamples)).astype(float)

        # Fill in the matrix
        for sigNo, (start, stop) in enumerate(zip(startIdx, endIdx)):
            out[sigNo, :] = sig[start:stop]

    return out
