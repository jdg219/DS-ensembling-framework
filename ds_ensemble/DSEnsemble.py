"""
This file holds the object that handles the 
DS Ensembling
"""
from pydantic.typing import List, Union, Tuple
from numpy import ndarray
import numpy as np
from pandas import get_dummies
import pyds

from ds_ensemble.Model import Model

class DSEnsemble():
    
    """
    This class will enable DS ensembling of models based 
    on user provided models and evaluation pipelines
    """
    def __init__(self, models: List[Model], evaluation_set: ndarray):
        
        # set the class attributes
        self.models = models
        self.evaluation_set = evaluation_set

        # create an empty attribute to hold belief results
        self.model_beliefs = list()
        self.__setup_bel_eval__()

    def __setup_bel_eval__(self):
        """
        This method generates beliefs for each class based on the
        passed in eval set, as sklearn models don't give 
        'belief' predictions, rather justa single class
        """
        for model in self.models:
            if model.type == 'sklearn':
                # get the predictions on the eval set
                preds = model.predict(self.evaluation_set[0])

                # group by the predicted classes and get belief for each
                # class based on pred
                counts = np.zeros([len(model.outputs), len(model.outputs)], dtype=np.float)
                for pred, truth in zip(preds, self.evaluation_set[1]):
                    counts[pred, truth] += 1


                # ensure small delta to never have 0 or 1 beliefs
                counts = np.where(counts==0, 0.1, counts)

                # normalize and add to beliefs
                normed = counts/counts.sum(axis=-1, keepdims=True)

                self.model_beliefs.append(normed)

            else:
                self.model_beliefs.append(np.zeros([self.num_output_classes, self.num_output_classes], dtype=np.float))
        
        # convert to nparray
        self.model_beliefs = np.array(self.model_beliefs)


    def predict(self, pred_data: ndarray, decision_metric:str='bel'):
        """
        Predict on the evaluation set for each model and 
        return the ensemble results
        """
        cumulative_beliefs = list()
        
        # for each model, predict on the results
        for i in range(len(self.models)):

            # get the relevant details for this iteration
            model = self.models[i]
            beliefs = self.model_beliefs[i,:,:]

            # predict on the data
            preds = model.predict(pred_data)

            # matrix lookup for beliefs corresponding to pred
            cur_bels = np.array([beliefs[class_pred,:] for class_pred in preds])

            # save off in our final array
            cumulative_beliefs.append(cur_bels)

        # now that we have all belies across the models, we can do ds ensembling
        # returned array will be n_samples x n_classes x 2 where the last dimension has
        # belief in the 0th index and plausibility in the 1th index
        ensembled_results = self.__dempster_combination__(np.array(cumulative_beliefs))

        # now we predict based on the selected method
        if decision_metric == 'bel':
            decision_data = np.squeeze(ensembled_results[:,:,0])
            preds = np.argmax(decision_data, axis=-1)

        return preds

    def __dempster_combination__(self, cumulative_beliefs: ndarray):
        """
        do dempster combination on the resulting belief matrix from previous predictions,
        input is n_models x n_samples x n_outputs and each entry is the corresponding belief
        """

        # loop over all samples
        for i in range(cumulative_beliefs.shape[1]):
            # samples will then be that index
            belief_entries = np.squeeze(cumulative_beliefs[:,i,:])
            # belief entries is now a n_models x n_output_classes
            # entry for the current sample

            # now we ensemble based on the classes for each model
            for j in range(len(self.models)):

                # construct the 

        # currently we can just dot across the last dimension and normalize,
        # since all is single dimensional then plaus = bel
        prod = np.ones([cumulative_beliefs.shape[0], cumulative_beliefs.shape[1]])
        for i in range(cumulative_beliefs.shape[-1]):
            prod = np.multiply(prod, np.squeeze(cumulative_beliefs[:,:,i]))

            # normalize to minimize underflow errors
            prod = prod/prod.sum(axis=-1, keepdims=True)

        # setup our return
        combined_results = np.zeros([cumulative_beliefs.shape[0], cumulative_beliefs.shape[1], 2],
                                    dtype=np.float)

        combined_results[:,:,0] = prod
        combined_results[:,:,1] = prod

        #TODO: expand this to work across multiple class overlaps eventually

        return combined_results

