"""
In this test we evaluate whether or not 
DS Ensembling will work as well as majority vote
when the models are quite dependent, specifically
when we have the same model just trained on different
crossval folds
"""

from sklearn.svm import SVC as svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

from numpy import ndarray
import numpy as np
import pandas as pd
from scipy.stats import mode

from ds_ensemble.DSEnsemble import DSEnsemble as DSE
from ds_ensemble.Model import Model

if __name__=='__main__':
    # load in the iris dataset
    (x,y) = load_breast_cancer(return_X_y=True)

    acc_dse = list()
    pr_dse = list()
    rec_dse = list()
    acc_mv = list()
    pr_mv = list()
    rec_mv = list()

    for w in range(500):
        print(w)

        # split our data into train test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        # now crossval on train
        cv = KFold(n_splits=5)
        folds = cv.split(x_train, y_train)

        # define list to hold classifiers
        clf_list = []

        # now iterate over each folds
        for val_ind, hold_ind in folds:

            # holdout an extra bit of data from train to use as the evaluation set
            # for DSE
            x_tr = x_train[val_ind]
            x_eval = x_train[hold_ind] 
            y_tr = y_train[val_ind]
            y_eval = y_train[hold_ind] 

            # create our classifiers
            clf_svm = svm()

            # fit our models now
            clf_svm.fit(x_tr, y_tr)
        

            # place them in the model wrapper
            svm_wrap = Model(trained_model = clf_svm,
                            model_type = 'sklearn',
                            output_classes = ['0', '1'],
                            preprocess_function = None
                            )
        

            # now setup the beliefs for each model based on our eval set
            svm_wrap.setup_beliefs([x_eval, y_eval])

            # add to the list
            clf_list.append(svm_wrap)

        # now we instantiate our ds ensembling
        dse = DSE(models=clf_list)

        # finally we can predict on our test set using belief as
        # the predictor method
        results = dse.predict(x_test, decision_metric='bel')

        # we can also now do majority vote from the classifiers
        indiv_preds = []
        for mdl in clf_list:
            preds = mdl.model.predict(x_test)
            indiv_preds.append(preds)

        stack = np.stack(indiv_preds, axis=1)
        maj_vote = mode(stack, axis=1).mode

        # acc and precision and recall arrays
        pr_dse.append(precision_score(results, y_test, average='weighted'))
        pr_mv.append(precision_score(maj_vote, y_test, average='weighted'))

        rec_dse.append(recall_score(results, y_test, average='weighted'))
        rec_mv.append(recall_score(maj_vote, y_test, average='weighted'))

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
    df.to_csv('crossfold_test_metrics.csv')
