import numpy as np
import pandas as pd
from src import config
from scipy.io import loadmat


def readData(filePath = config.INDEX_FILE):
    ''' Reads the index file '''

    # Read index
    df = pd.melt(frame      = pd.read_csv(config.INDEX_FILE, sep = ';'), 
                 id_vars    = ['Folder', 'FaultDiameter_inches', 'MotorLoad_HP', 'MotorSpeed_rpm'],
                 var_name   = 'FaultType',
                 value_name = 'matFile')

    # Drop rows with no corresponding mat. file
    df = df[~df['matFile'].isnull()]
    
    # Assign 0 values to fault diameter for files with no faults
    rowIndex = df['Folder'] == 'NormalBaseline'
    df.loc[rowIndex, 'FaultDiameter_inches'] = 0

    # Convert matFile name to int
    df['matFile'] = df['matFile'].astype(int)
    
    # Make path for each file
    paths = config.DATA_PATH + df['matFile'].astype(str) + '.mat'
    paths[df['matFile'].isnull()] = None # 'None' paths for non-existing matlab files
    
    # Read vibration files
    df['FanVibs'], df['DriveVibs'] = _parseFiles(paths)
    
    # Not needed anymore
    df.drop(['matFile'], axis = 1, inplace = True)
    
    # Reset index
    df.index = np.arange(df.shape[0])
    
    # Make labels
    df['Label'] = _makeLabels(folderList    = df['Folder'].values, 
                              faultTypeList = df['FaultType'].values)
    
    return df


def _parseFiles(pathFiles):
    ''' Calls _parseFile() over a list of files '''
    
    fanVibs, driveVibs = [], []

    for file in pathFiles:
        fanVib, driveVib = _parseFile(file)
        fanVibs.append(fanVib)
        driveVibs.append(driveVib)
        
    return fanVibs, driveVibs


def _parseFile(filePath):
    ''' Parses a single matlab file '''
    
    # Load file
    data = loadmat(filePath)

    # Grab variable names without headers
    varNames = [key for key in data.keys() if ('DE_time' in key or 'FE_time' in key)]
    noVars   = len(varNames)


    if noVars > 2: # More than (the expected) two vibrations signals were found 

        fileNo = filePath.split('/')[-1].strip('.mat') # Grab filename from filepath

        # Two digit file -> add leading zero
        if len(fileNo) < 3: fileNo = '0' + fileNo

        # It's necessary to specify exactly which variables to read as some files contain 
        # variables from multiple experiments
        fanEndVibrationVar   = 'X' + fileNo + '_FE_time'
        driveEndVibrationVar = 'X' + fileNo + '_DE_time'

        fanEndData   = np.squeeze(data[fanEndVibrationVar])
        driveEndData = np.squeeze(data[driveEndVibrationVar])

    elif noVars < 2: # Less than two vibration signals were found

        varNames = varNames[0] # Unpack
        driveEndData = None
        fanEndData   = None
        if   'DE_time' in varNames: driveEndData = np.squeeze(data[varNames])
        elif 'FE_time' in varNames: fanEndData   = np.squeeze(data[varNames])

    else: # Exactly two files were found

        for varName in varNames:
            if   'DE_time' in varName: driveEndData = np.squeeze(data[varName])
            elif 'FE_time' in varName: fanEndData   = np.squeeze(data[varName])
            
    return fanEndData, driveEndData


def _makeLabels(folderList, faultTypeList):
    ''' Generates the classification labels for the dataset '''
    
    # make label = location (drive/fan end) + diameter + type
    faultLocation = [elem.split('_')[0].replace('End', '') for elem in folderList]

    # Remove position from fault type
    subStrings = ['Centered', 'Orthogonal', 'Opposite']
    faultType  = [_replaceMultiple(elem.split('_')[0], subStrings, '') for elem in faultTypeList]

    # Join strings
    labels = ["{}_{}".format(elem1, elem2) for elem1, elem2 in zip(faultLocation, faultType)]
    
    return labels


def _replaceMultiple(string, oldSubstringList, newSubstring):
    ''' Replaces multiple substrings with a new substring '''
    
    for oldSubstring in oldSubstringList:
        string = string.replace(oldSubstring, newSubstring)
        
    return string