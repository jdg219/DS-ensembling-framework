"""
In this demo we move away from the iris dataset
and instead look at a toy dataset where we have points in the 2d plane.
Each quadrant then has its own class. We demonstrate how 
DS can use 2 linear classifiers to solve the task while otherwise
we could not with multiple of the weak classifiers in a standard ensemble
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

from ds_ensemble.DSEnsemble import DSEnsemble as DSE
from ds_ensemble.Model import Model

if __name__=='__main__':
    # load in the quadrant
    df = pd.read_csv('quadrant_dataset.csv')
    x = df[['x_points','y_points']]
    y = df.labels
    

    # split our data into holdout/train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # holdout an extra bit of data from train to use as the evaluation set
    # for DSE
    x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, test_size=0.2)

    # create our two classifiers
    clf_lr1 = LogisticRegression()
    clf_lr2 = LogisticRegression()

    # now we artificially combine classes 0&1 vs 2&3 and for for clf1
    # and 1 and 2 to be 1 for the KNN
    y_train_lr1 = np.where((y_train==0) | (y_train==1), 0, 1)
    y_eval_lr1 = np.where((y_eval==0) | (y_eval==1), 0, 1)

    # now we split 0&3 vs 1&2 for lr2
    y_train_lr2 = np.where((y_train==0) | (y_train==3), 0, 1)
    y_eval_lr2 = np.where((y_eval==0) | (y_eval==3), 0, 1)

    # fit our models now
    clf_lr1.fit(x_train, y_train_lr1)
    clf_lr2.fit(x_train, y_train_lr2)

    # place them in the model wrapper
    lr1_wrap = Model(trained_model = clf_lr1,
                     model_type = 'sklearn',
                     output_classes = ['01', '23'],
                     preprocess_function = None
                    )
    
    lr2_wrap = Model(trained_model = clf_lr2,
                     model_type = 'sklearn',
                     output_classes = ['03', '12'],
                     preprocess_function = None
                    )

    # now setup the beliefs for each model based on our eval set
    lr1_wrap.setup_beliefs([x_eval, y_eval_lr1])
    lr2_wrap.setup_beliefs([x_eval, y_eval_lr2])

    # now we instantiate our ds ensembling
    dse = DSE(models=[lr1_wrap, lr2_wrap])
    
    # finally we can predict on our test set using belief as
    # the predictor method
    results = dse.predict(x_test, decision_metric='bel')

    # see the accuracy of the results
    cf_matrix = confusion_matrix(results, y_test)
    print(cf_matrix)