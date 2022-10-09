"""
This file holds the object that handles the 
DS Ensembling
"""
from pydantic.typing import List, Union
from numpy import ndarray
import numpy as np
from pandas import get_dummies

def DSEnsemble():
    
    """
    This class will enable DS ensembling of models based 
    on user provided models and evaluation pipelines
    """
    def __init__(self, models: List, model_types:Union[List[str], str],
                holdout_set: tuple(ndarray, ndarray), output_class_count: int,
                preprocess_functions=None):
        
        # set the class attributes
        self.models = models
        self.model_types = model_types
        self.holdout_set = holdout_set
        if preprocess_functions:
            self.preprocess_functions = preprocess_functions
        else:
            self.preprocess_functions = [None for model in models]
        self.num_output_classes = output_class_count

        # create an empty attribute to hold belief results
        self.model_beliefs: List[ndarray] = self.__setup_bel_eval__()

    def __setup_bel_eval__(self):
        """
        This method generates beliefs for each class based on the
        passed in eval set
        """
        for model in self.models:
            self.model_beliefs.append(np.ones([len(np.unique(self.num_output_classes))], dtype=np.float))

    def predict(self, pred_data: ndarray, decision_metric:str='bel'):
        """
        Predict on the evaluation set for each model and 
        return the ensemble results
        """
        # preallocate array to hold belief outputs in array
        # of shape: n_samples x n_classes x n_models
        cumulative_beliefs = np.zeros([pred_data.shape[0], self.num_output_classes, len(self.models)])

        # for each model, predict on the results
        for i in range(len(self.models)):

            # get the relevant details for this iteration
            model = self.models[i]
            norm_func = self.preprocess_functions[i]
            beliefs = self.model_beliefs[i]

            # apply the normalization if relevant
            if norm_func:
                pred_data = norm_func(pred_data)

            # predict on the data
            preds = model.predict(pred_data)

            # convert it to beliefs
            # first need one hot preds for calculations
            preds_oh = pd.get_dummies(preds)
            # matrix multiply for beliefs
            cur_bels = np.matmul(preds_oh, beliefs)

            # save off in our final array
            cumulative_beliefs[:,:,i] = cur_bels

        # now that we have all belies across the models, we can do ds ensembling
        # returned array will be n_samples x n_classes x 2 where the last dimension has
        # belief in the 0th index and plausibility in the 1th index
        ensembled_results = self.__dempster_combination__(cumulative_beliefs)

        # now we predict based on the selected method
        if decision_metric == 'bel':
            decision_data = np.squeeze(ensembled_results[:,:,0])
            preds = np.argmax(decision_data, axis=-1)

        return preds

    def __dempster_combination__(self, cumulative_beliefs: ndarray):
        #TODO: Actually make this work correctly
        # for now we just average
        combined_results = np.zeros([cumulative_beliefs.shape[0], cumulative_beliefs.shape[1], 2],
                                    dtype=np.float)

        summed = np.sum(combined_results, axis=-1)
        combined_results[:,:,0] = summed/np.sum(summed, axis=-1)
        combined_results[:,:,1] = 1 - combined_results[:,:,0]

        return combined_results

