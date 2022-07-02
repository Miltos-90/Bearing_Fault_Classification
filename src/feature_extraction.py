from scipy import stats, signal
from collections import defaultdict
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd

from src import config
from src.FFT import FFTAnalysis as FFT


def _extractTimeDomainFeatures(sig):
    ''' Extracts time domain features from one vibration signal'''

    # Get time features
    features = dict()
    
    if sig is not None:
        
        rmsVal  = np.sqrt(np.square(sig).mean())
        sigMax  = sig.max()
        absSig  = np.abs(sig)
        absMean = absSig.mean()
    
        features = {
            'mean':     sig.mean(), 
            'min':      sig.min(),
            'max':      sigMax,
            'std':      np.std(sig),
            'skew':     stats.skew(sig),
            'kurt':     stats.kurtosis(sig),
            'rms':      rmsVal,
            'p2p':      sigMax - sig.min(),
            'crestF':   sigMax / rmsVal,
            'impulseF': absSig.max() / absMean,
            'shapeF':   rmsVal / absMean
            }
        
    return features


def _getSubBand(FFTfreqs, FFTlevels, band):
    ''''Extract portion of the FFT corresponding to the given frequency band '''
    
    idxKeep = (FFTfreqs >= band[0]) & (FFTfreqs < band[1])
    freqs   = FFTfreqs[idxKeep]
    levels  = FFTlevels[idxKeep]
    
    return freqs, levels


def _extractPeaks(freqs, levels, distance):
    ''' Extracts peaks from an FFT '''
    
    peakIdx, _ = signal.find_peaks(levels, distance = distance)
    peakLevels = levels[peakIdx]
    peakFreqs  = freqs[peakIdx]

    return peakFreqs, peakLevels


def _getTopPeaks(levels, freqs, noPeaks):
    ''' Extract top <n> peaks from an FFT '''

    # Sort peak indices from highest to lowest level
    sortIdx = np.argsort(levels)
    sortIdx = np.flip(sortIdx)
    
    # Grab top <n> peaks
    levels = levels[sortIdx[0:noPeaks]]
    freqs  = freqs[sortIdx[0:noPeaks]]
    
    return freqs, levels
    

def _extractFrequencyDomainFeatures(sig, FFTsettings,
                                    noPeaks  = config.FBAND_PEAKS,
                                    peakDist = config.PEAK_DIST,
                                    fs       = config.Fs, 
                                    fBands   = config.FBANDS):
    ''' Extracts frequency domain features from one vibration signal'''
                   
    FFTfreqs, FFTlevels = None, None
    features = defaultdict()

    if sig is not None:
        FFTfreqs, FFTlevels = FFT(sig, config.Fs, FFTsettings)
    
        # Split in bands
        for bandNo, band in enumerate(fBands):

            freqs, levels = _getSubBand(FFTfreqs, FFTlevels, band)
            freqs, levels = _extractPeaks(freqs, levels, peakDist)
            freqs, levels = _getTopPeaks(levels, freqs, noPeaks)

            # Add peaks from current band to the dictionary with the features
            for peakNo in range(noPeaks):
                featName           = f'band_{bandNo + 1}_peak_{peakNo+1}_level'
                features[featName] = levels[peakNo]
                featName           = f'band_{bandNo + 1}_peak_{peakNo+1}_freq'
                features[featName] = freqs[peakNo]
            
    return features


def _extractFeatures(sig, FFTsettings):
    ''''Extracts time- and frequency-domain features from one vibration signal'''
    
    feats     = _extractTimeDomainFeatures(sig)
    freqFeats = _extractFrequencyDomainFeatures(sig, FFTsettings)
    feats.update(freqFeats)
    
    return feats


def extractDatasetFeatures(df, FFTsettings = config.FFT_SETTINGS):
    ''' Extracts features from the entire dataset '''

    # Extract features from every experiment
    driveFeats, fanFeats = [], []
    for idx, record in tqdm(df.iterrows(), total = df.shape[0]):

        FFTsettings['HPCutoffFrequency'] = 20 * record['MotorSpeed_rpm'] / 60
        FFTsettings['LPCutoffFrequency'] = 10 * record['MotorSpeed_rpm'] / 60 

        driveFeats.append(_extractFeatures(record['DriveVibs'], FFTsettings))
        fanFeats.append(_extractFeatures(record['FanVibs'], FFTsettings))

    # Make dataframes with the extracted features
    dfDrive = pd.DataFrame(driveFeats)
    dfFan   = pd.DataFrame(fanFeats)

    # Add corresponding labels
    dfDrive['label'] = df['DriveLabel']
    dfFan['label']   = df['FanLabel']

    # Remove rows with missing records for fan-end bearing
    dfFan.dropna(axis = 0, how = 'any', inplace = True)
    
    return dfDrive, dfFan
    