"""
This file tests performance of the ensembling methods when a classifier
is good and proportionally more are bad
"""


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC as svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

import numpy as np
import pandas as pd
from scipy.stats import mode

from ds_ensemble.DSEnsemble import DSEnsemble as DSE
from ds_ensemble.Model import Model


if __name__=='__main__':
    # load in the hastie data 
    (x,y) = make_gaussian_quantiles(n_samples=500, n_classes=5, n_features=5)

    acc_dse = list()
    pr_dse = list()
    rec_dse = list()
    acc_mv = list()
    pr_mv = list()
    rec_mv = list()

    for w in range(500):
        print(w)

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

        # we can also now do majority vote from the classifiers
        indiv_preds = []
        for mdl in mdl_list:
            preds = mdl.model.predict(x_test)
            indiv_preds.append(preds)

        stack = np.stack(indiv_preds, axis=1)
        maj_vote = mode(stack, axis=1).mode

        # acc and precision and recall arrays
        pr_dse.append(precision_score(results, y_test, average='weighted', zero_division=0))
        pr_mv.append(precision_score(maj_vote, y_test, average='weighted', zero_division=0))

        rec_dse.append(recall_score(results, y_test, average='weighted', zero_division=0))
        rec_mv.append(recall_score(maj_vote, y_test, average='weighted', zero_division=0))

        acc_dse.append(accuracy_score(results, y_test))
        acc_mv.append(accuracy_score(maj_vote, y_test))

    # finally save off in df
    df_dict = {
        'run_number': np.arange(500),
        'DS_precision': pr_dse,
        'DS_recall': rec_dse,
        'DS_acc': acc_dse,
        'MV_precision': pr_mv,
        'MV_recall': rec_mv,
        'MV_acc': acc_mv
    }
    df = pd.DataFrame(df_dict)
    df.to_csv('good_and_poor_test_metrics.csv')