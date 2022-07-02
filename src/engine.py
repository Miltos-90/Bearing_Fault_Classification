import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from tqdm.notebook import tqdm
import lightgbm as lgb
from src import config

'''Performs nested kFold CV with random search'''
def nestedKFoldCV(X, y, 
                  estimator  = lgb.LGBMClassifier(random_state = config.SEED),
                  score      = config.SCORE,
                  parameters = config.PARAMETERS, 
                  noFolds    = config.NO_FOLDS, 
                  noSearch   = config.NO_SEARCH, 
                  noParallel = config.NO_PARALLEL,
                  seed       = config.SEED):
    
    skf   = StratifiedKFold(n_splits = noFolds, shuffle = True, random_state = seed)
    yhat  = np.zeros_like(y)
    iter_ = tqdm(enumerate(skf.split(X, y)), total = noFolds)
    
    for foldNo, (train_idx, test_idx) in iter_:

        # Train/test split
        Xtrain, ytrain = X[train_idx, :], y[train_idx]
        Xtest, ytest   = X[test_idx, :],  y[test_idx]

        # Run random search with kFold CV
        if noSearch > 0:
            clf = RandomizedSearchCV(estimator           = estimator,
                                     cv                  = StratifiedKFold(n_splits = noFolds - 1, shuffle = True, random_state = seed),
                                     param_distributions = parameters, 
                                     n_iter              = noSearch, 
                                     n_jobs              = noParallel,
                                     scoring             = 'roc_auc_ovr_weighted', 
                                     random_state        = seed).fit(Xtrain, ytrain)
        else:
            clf = estimator
        
        # Fit estimator
        clf.fit(Xtrain, ytrain)
        
        # Get predictions on the test set
        yhat[test_idx] = clf.predict(Xtest).astype(int)

    return yhat