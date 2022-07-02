import pandas as pd
import numpy as np
from src import config
from src.FFT import _ButterFilter as bfilter


def _downsample(sig, samplingFrequency = 48000, downsampleRatio = 4):
    ''' Downsamples a signal to a selected ratio'''
    
    # Compute cutoff frequency
    cutoffFrequency = (samplingFrequency / downsampleRatio) / (2.56)
    
    # Low-pass filter the input signal
    sig = bfilter(sig, samplingFrequency,
                  cutoffFrequency = cutoffFrequency,
                  order           = 4,
                  filterType      = 'LP')

    # Skip indices
    idxToKeep = np.arange(start = 0, stop = sig.shape[0], step = downsampleRatio)
    sig       = sig[idxToKeep]
    
    return sig

def makeBearingLabels(labelList):
    ''' Generates labels for each bearing based on the dataframe labels'''
    
    driveLabels = []
    fanLabels   = []
    for label in labelList:
        if ('Normal' in label) or ('Fan' in label):
            driveLabels.append('Normal')
        else:
            driveLabels.append(label)

        if ('Normal' in label) or ('Drive' in label):
            fanLabels.append('Normal')
        else:
            fanLabels.append(label)
        
    return driveLabels, fanLabels
        

def _chunks(sig, n):
    """Yield successive n-sized chunks from a 1D numpy array."""
    
    for i in range(0, len(sig), n):
        yield sig[i:i + n]


def _chunkSignal(sig, noSamples):
    ''' Chunk a signal in a list of signals containing <noSamples> samples '''
    
    if sig is not None:
        chunkedSig = []
        for chunk in _chunks(sig, noSamples):

            if chunk.shape[0] == noSamples: # Skip the last samples of the signal
                chunkedSig.append(chunk)
    else:
        chunkedSig = []
    
    return chunkedSig

def resampleVibrations(df, noSamples):
    ''' Resample the vibration signals to have the same number of records each (10 sec) '''
    
    newRecords = []
    for idx, record in df.iterrows():

        driveSigChunks = _chunkSignal(record['DriveVibs'], noSamples)
        fanSigChunks   = _chunkSignal(record['FanVibs'],   noSamples)

        # Some files do not contain fan end records of vibration. Make a list of None for those
        if not fanSigChunks: fanSigChunks = [None] * len(driveSigChunks)

        for driveSig, fanSig in zip(driveSigChunks, fanSigChunks):
            newRecord = record.copy()
            newRecord['DriveVibs'] = driveSig
            newRecord['FanVibs']   = fanSig

            newRecords.append(newRecord)

    return pd.DataFrame.from_records(newRecords)


def downsample48kHzVibrations(df):
    ''' Downsamples vibrations from 48kHz to 12 kHz '''
    fanVibs, driveVibs = [], []

    for idx, record in df.iterrows():

        is48kHz = '48k' in record['Folder']

        if is48kHz:
            fanVib   = _downsample(record['FanVibs'], samplingFrequency = 48000, downsampleRatio = 4)
            driveVib = _downsample(record['DriveVibs'], samplingFrequency = 48000, downsampleRatio = 4)
        else:
            fanVib   = record['FanVibs']
            driveVib = record['DriveVibs']

        fanVibs.append(fanVib)
        driveVibs.append(driveVib)

    df['FanVibs']   = fanVibs
    df['DriveVibs'] = driveVibs
    
    return df