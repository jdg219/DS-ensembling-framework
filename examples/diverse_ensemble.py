"""
Demo the ensembling of more diverse models
as well as the case where we have different subsets 
of the output space covered by models

Ex:
NN - predicts all output classes
&
KNN - predicts classes 1 vs 2&3
"""

import tensorflow as tf
import xgboost as xgb

from ds_ensemble.DSEnsemble import DSEnsemble
from ds_ensemble.Model import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from numpy import ndarray
import numpy as np
import pandas as pd

# create some normalization functions to help
# demo data transforms for ensembling
def normalize_iris_data(unnorm_data: ndarray):
    # for each feature in the dataset, make the values 
    # 0 mean 1 std
    # we copy to not affect the passed in data
    norm_data = unnorm_data.copy()

    for feat_num in range(unnorm_data.shape[1]):
        # get the mean and var
        mean = np.mean(unnorm_data[:][feat_num])
        std = np.std(unnorm_data[:][feat_num])

        # apply to the data in place
        norm_data[:][feat_num] = (unnorm_data[:][feat_num] - mean) / float(std)

    # return the normalize array
    return norm_data

if __name__=='__main__':
    # load in the iris dataset
    (x,y) = load_iris(return_X_y=True)

    # split our data into holdout/train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # holdout an extra bit of data from train to use as the evaluation set
    # for DSE
    x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, test_size=0.2)

    # normalize x for training
    x_train_norm = normalize_iris_data(x_train)

    # make knn predict not full output space
    # only class 0 vs 1&2
    y_train_knn = np.where(y_train==0, 0, 1)
    y_eval_knn = np.where(y_eval==0, 0, 1)

    # make one hot for nn
    y_train_nn = pd.get_dummies(y_train).values

    # create our classifiers
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_xg = xgb.XGBClassifier(n_estimators=25)
    clf_nn = tf.keras.models.Sequential([
                        tf.keras.layers.Dense(10, activation='relu'),
			tf.keras.layers.Dense(5, activation='relu'),
			tf.keras.layers.Dense(3, activation='softmax')
		])
    clf_nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # train/fit each
    clf_knn.fit(x_train_norm, y_train_knn)
    clf_xg.fit(x_train_norm, y_train)
    clf_nn.fit(x_train_norm, y_train_nn, batch_size=25, epochs=100)

    # wrap in models
    knn_wrap = Model(trained_model = clf_knn,
                     model_type = 'sklearn',
                     output_classes = ['0', '12'],
                     preprocess_function = normalize_iris_data
                    )
    
    xg_wrap = Model(trained_model = clf_xg,
                     model_type = 'xgboost',
                     output_classes = ['0', '1', '2'],
                     preprocess_function = normalize_iris_data
                    )

    nn_wrap = Model(trained_model = clf_nn,
                     model_type = 'tensorflow',
                     output_classes = ['0', '1', '2'],
                     preprocess_function = normalize_iris_data
                    )

    # setup the beliefs for each
    knn_wrap.setup_beliefs([x_eval, y_eval_knn])
    xg_wrap.setup_beliefs([x_eval, y_eval])
    nn_wrap.setup_beliefs([x_eval, y_eval])

    # now we instantiate our ds ensembling
    dse = DSEnsemble(models=[xg_wrap, knn_wrap, nn_wrap])

    # finally we can predict on our test set using avg of bel and pr as
    # the predictor method
    results = dse.predict(x_test, decision_metric='avg_bel_prec')
    for pred, true in zip(results, y_test):
        print("Pred: ", pred, ", True: ", true)


    # see the accuracy of the results
    cf_matrix = confusion_matrix(results, y_test)
    print(cf_matrix)
    

