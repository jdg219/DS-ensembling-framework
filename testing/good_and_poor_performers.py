"""
This file tests performance of the ensembling methods when a classifier
is good and proportionally more are bad
"""


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC as svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, f1_score

import numpy as np
from scipy.stats import mode

from ds_ensemble.DSEnsemble import DSEnsemble as DSE
from ds_ensemble.Model import Model


if __name__=='__main__':
    # load in the hastie data 
    (x,y) = make_gaussian_quantiles(n_samples=500, n_classes=5, n_features=5)

    # split our data into holdout/train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # create our two classifiers
    clf_svm = svm()
    clf_knn1 = LogisticRegression()
    clf_knn2 = LogisticRegression()
    clf_knn3 = LogisticRegression()
    clf_knn4 = LogisticRegression()

    # train across folds for each model
    clf_list = [clf_svm, clf_knn1, clf_knn2, clf_knn3, clf_knn4]

    # now crossval on train
    cv = KFold(n_splits=5)
    folds = cv.split(x_train, y_train)

    # output list of model classes
    mdl_list = []

    for clf, (val_ind, hold_ind) in zip(clf_list, folds):

        # holdout an extra bit of data from train to use as the evaluation set
        # for DSE
        x_tr = x_train[val_ind]
        x_eval = x_train[hold_ind] 
        y_tr = y_train[val_ind]
        y_eval = y_train[hold_ind] 

        # fit our models now
        clf.fit(x_tr, y_tr)

        # place them in the model wrapper
        mdl_list.append(Model(trained_model = clf,
                        model_type = 'sklearn',
                        output_classes = ['0', '1', '2', '3', '4'],
                        preprocess_function = None
                        ))

        # now setup the beliefs 
        mdl_list[-1].setup_beliefs([x_eval, y_eval])

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
        preds = mdl.model.predict(x_test)
        indiv_preds.append(preds)

    stack = np.stack(indiv_preds, axis=1)
    maj_vote = mode(stack, axis=-1, keepdims=False).mode

    # output the ensemble results
    cf_matrix_2 = confusion_matrix(maj_vote, y_test)
    print("Ensemble Confusion Matrix\n\n", cf_matrix_2)

    print(f"DSE F1 score: {f1_score(results, y_test, average='weighted')}")
    print(f"Ensemble F1 Score: {f1_score(maj_vote, y_test, average='weighted')}")

