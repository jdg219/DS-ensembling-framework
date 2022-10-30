"""
This file will demonstrate a basic comparison between ensembling
and DS combination
"""

from sklearn.svm import SVC as svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

from numpy import ndarray
import numpy as np
import pandas as pd
from scipy.stats import mode

from ds_ensemble.DSEnsemble import DSEnsemble as DSE
from ds_ensemble.Model import Model

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

        # holdout an extra bit of data from train to use as the evaluation set
        # for DSE
        x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, test_size=0.2)

        # create our classifiers
        clf_svm = svm()
        clf_knn = KNeighborsClassifier(n_neighbors=3)
        clf_lr = LogisticRegression(multi_class='ovr', solver='liblinear')

        # normalize knn and log reg
        x_train_norm = normalize_iris_data(x_train)

        # fit our models now
        clf_svm.fit(x_train, y_train)
        clf_knn.fit(x_train_norm, y_train)
        clf_lr.fit(x_train_norm, y_train)

        # place them in the model wrapper
        svm_wrap = Model(trained_model = clf_svm,
                        model_type = 'sklearn',
                        output_classes = ['0', '1', '2'],
                        preprocess_function = None
                        )
        
        knn_wrap = Model(trained_model = clf_knn,
                        model_type = 'sklearn',
                        output_classes = ['0', '1', '2'],
                        preprocess_function = normalize_iris_data
                        )

        lr_wrap = Model(trained_model = clf_lr,
                        model_type = 'sklearn',
                        output_classes = ['0', '1', '2'],
                        preprocess_function = normalize_iris_data
                        )

        # now setup the beliefs for each model based on our eval set
        svm_wrap.setup_beliefs([x_eval, y_eval])
        knn_wrap.setup_beliefs([x_eval, y_eval])
        lr_wrap.setup_beliefs([x_eval, y_eval])

        # now we instantiate our ds ensembling
        dse = DSE(models=[svm_wrap, knn_wrap])

        # finally we can predict on our test set using belief as
        # the predictor method
        results = dse.predict(x_test, decision_metric='bel')

        # we can also now do majority vote from the classifiers
        pred_svm = clf_svm.predict(x_test)
        pred_knn = clf_knn.predict(normalize_iris_data(x_test))
        pred_lr = clf_lr.predict(normalize_iris_data(x_test))
        stack = np.stack([pred_svm, pred_knn, pred_lr], axis=1)
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
    df.to_csv('basic_test_metrics.csv')