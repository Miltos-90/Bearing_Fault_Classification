import numpy as np

DATA_PATH   = 'C:/Users/kalika01/Desktop/CWRU Bearing Dataset/data/matFiles/' # Path to folder containing all data
INDEX_FILE  = 'C:/Users/kalika01/Desktop/CWRU Bearing Dataset/data/index.csv' # Path to index file

FBANDS      = [(0, 150),              # Frequency bands from which FFT features will be separately extracted
               (150, 300), 
               (300, 1500), 
               (1500, 99999)]
Fs          = 12000                   # Sampling frequency
PEAK_DIST   = 10                      # Distance [points] between consecutive peaks from each sub-band of the frequency domain
FBAND_PEAKS = 2                       # No. peaks to extract from each sub-band of the frequency domain
NO_FOLDS    = 10                      # No. folds for Nested kfold CV
NO_SEARCH   = 1000                    # No. iterations for random search
SEED        = 132                     # Seed for training
NO_PARALLEL = 10                      # Number of jobs to run in parallel
SCORE       = 'roc_auc_ovr_weighted'  # Strategy to evaluate the performance of the cross-validated model on the validation set.

FFT_SETTINGS = {                      # Dictionary with FFT settings
    'analysisType'      : 'envelope', # regular
    'HPFilterOrder'     : 1,          # Envelope (HP) Filter
    'FFTstride'         : 2 ** 16,    # try the highest power of 2 that fits the signal length. Treat this number as an upper limit
    'FFToverlap'        : 0.75,
    'FFTwindowName'     : 'hann',
    'FFTavgType'        : 'maxhold',  # For regular analysis set to 'linear'
                                      # For envelope set to 'maxhold'
    'LPFilterOrder'     : 1,          # LP filter (used only in envelope analysis)
}

PARAMETERS = dict(                    # Parameters for random search /w LightGBM
    boosting_type    = ['dart'],
    reg_alpha        = list(10 ** np.arange(-6, 4, 0.2)),
    reg_lambda       = list(10 ** np.arange(-6, 4, 0.2)),
    n_estimators     = [20, 40, 80, 160, 320, 640, 1024],
    learning_rate    = [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08],
    max_depth        = np.arange(10, 40, 5),
    min_split_gain   = [0, 0.001,0.005,0.01],
    subsample        = [0.4, 0.5, 0.7, 0.9, 1.0],
    colsample_bytree = [0.1, 0.3, 0.5, 0.75, 1.0],
    num_leaves       = [5, 10, 20, 40, 80, 100, 200, 400, 800, 1000, 2000, 4000, 10000]
)
