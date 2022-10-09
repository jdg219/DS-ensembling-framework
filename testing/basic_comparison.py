"""
This file will demonstrate a basic comparison between ensembling
and DS combination
"""

from sklearn.svm import SVC as svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from numpy import ndarray
import numpy as np
from scipy.stats import mode

from ds_ensemble.DSEnsemble import DSEnsemble as DSE

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

    # now we instantiate our ds ensembling
    dse = DSE(  models=[clf_svm, clf_knn, clf_lr],
                model_types=['sklearn', 'sklearn', 'sklearn'],
                evaluation_set=(x_eval, y_eval),
                output_class_count = 3,
                preprocess_functions=[None, normalize_iris_data, normalize_iris_data]
            )

    # finally we can predict on our test set using belief as
    # the predictor method
    results = dse.predict(x_test, decision_metric='bel')

    # see the accuracy of the DS results
    cf_matrix = confusion_matrix(results, y_test)
    print("DS Result Confusion Matrix\n\n", cf_matrix)

    # we can also now do majority vote from the classifiers
    pred_svm = clf_svm.predict(x_test)
    pred_knn = clf_knn.predict(normalize_iris_data(x_test))
    pred_lr = clf_lr.predict(normalize_iris_data(x_test))
    stack = np.stack([pred_svm, pred_knn, pred_lr], axis=1)
    maj_vote = mode(stack, axis=-1, keepdims=False).mode

    # output the ensemble results
    cf_matrix_2 = confusion_matrix(maj_vote, y_test)
    print("Ensemble Confusion Matrix\n\n", cf_matrix_2)