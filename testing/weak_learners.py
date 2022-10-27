"""
This file tests performance of the ensembling methods when
we have "weak" classifiers and ensemble their results where
each is predicting one vs rest
"""


import tensorflow as tf
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, f1_score

import numpy as np
from scipy.stats import mode

from ds_ensemble.DSEnsemble import DSEnsemble as DSE
from ds_ensemble.Model import Model


if __name__=='__main__':
    # load in the iris dataset
    (x,y) = make_gaussian_quantiles(n_samples=500, n_classes=5, n_features=5)

    # split our data into holdout/train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # now crossval on train
    cv = KFold(n_splits=5)
    folds = cv.split(x_train, y_train)

    # output list of model classes
    mdl_list = []

    # helper list for later
    classes = '01234'

    for i, (val_ind, hold_ind) in enumerate(folds):

        # holdout an extra bit of data from train to use as the evaluation set
        # for DSE
        x_tr = x_train[val_ind]
        x_eval = x_train[hold_ind] 
        y_tr = y_train[val_ind]
        y_eval = y_train[hold_ind] 

        # make us predict 1 vs rest for each classifier as the "weak"
        # learner
        y_tr = np.where(y_tr==i, 1, 0)
        y_eval = np.where(y_eval==i, 1, 0)

        # create the classifier and fit it
        clf = tf.keras.models.Sequential([
                            tf.keras.layers.Dense(10, activation='relu'),
        		tf.keras.layers.Dense(5, activation='relu'),
        		tf.keras.layers.Dense(3, activation='softmax')
        	])
        clf_nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        clf.fit(x_train_norm, y_train_nn, batch_size=50, epochs=100)

        # wrap in models
        clf_wrap = Model(trained_model = clf,
                        model_type = 'tensorflow',
                        output_classes = [classes.replace(f'{i}',''), f'{i}'],
                        preprocess_function = None
                        )

        # setup the beliefs
        clf_wrap.setup_beliefs([x_eval, y_eval])

        # add to model list
        mdl_list.append(clf_wrap)

    # now we instantiate our ds ensembling
    dse = DSE(models=mdl_list)

    # finally we can predict on our test set using belief as
    # the predictor method
    results = dse.predict(x_test, decision_metric='bel')

    # see the accuracy of the DS results
    cf_matrix = confusion_matrix(results, y_test)
    print("DS Result Confusion Matrix\n\n", cf_matrix)

    # we can also now do majority vote from the classifiers
    indiv_preds = []
    for mdl in mdl_list:
        # select the prediction of class of interest instead of "rest"
        preds = mdl.model.predict(x_test)[:,1]
        indiv_preds.append(preds)

    stack = np.stack(indiv_preds, axis=1)
    maj_vote = np.argmax(stack, axis=-1, keepdims=False)

    # output the ensemble results
    cf_matrix_2 = confusion_matrix(maj_vote, y_test)
    print("Ensemble Confusion Matrix\n\n", cf_matrix_2)

    print(f"DSE F1 score: {f1_score(results, y_test, average='weighted')}")
    print(f"Ensemble F1 Score: {f1_score(maj_vote, y_test, average='weighted')}")
