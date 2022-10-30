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
    def __init__(self, models: List[Model]):
        
        # set the class attributes
        self.models = models

        # create an empty attribute to hold belief results
        self.model_beliefs = [model.belief for model in models]

    def predict(self, pred_data: ndarray, decision_metric:str='bel'):
        """
        Predict on the evaluation set for each model and 
        return the ensemble results
        """
        cumulative_beliefs = np.empty(len(self.models), dtype=object)
        
        # for each model, predict on the results
        for i in range(len(self.models)):

            # get the relevant details for this iteration
            model = self.models[i]
            beliefs = self.model_beliefs[i]

            # predict on the data
            preds = model.predict(pred_data)

            # matrix lookup for beliefs corresponding to pred
            cur_bels = np.array([beliefs[class_pred,:] for class_pred in preds])

            # save off in our final array
            cumulative_beliefs[i] = cur_bels

        # now that we have all belies across the models, we can do ds ensembling
        # returned array will be ensembled results for each sample
        ensembled_results = self.__dempster_combination__(cumulative_beliefs)

        # now we predict based on the selected method
        output = np.zeros(pred_data.shape[0], dtype=int)
        if decision_metric.lower() == 'bel':
            
            # get the max belief set of each sample
            for i in range(len(ensembled_results)):
                res, = ensembled_results[i].max_bel()
                output[i] = int(res)

        return output

    def __dempster_combination__(self, cumulative_beliefs: ndarray):
        """
        do dempster combination on the resulting belief matrix from previous predictions,
        input is n_models x n_samples x n_outputs and each entry is the corresponding belief
        """

        # loop over all samples
        # and create a DS combination result for each
        results = []
        for i in range(cumulative_beliefs[0].shape[0]):

            # now we ensemble based on the classes for each model
            bpas = []
            for j in range(len(self.models)):

                # extract the desired entries
                belief_entries = np.squeeze(cumulative_beliefs[j][i,:])

                # construct the bpas
                bpa = pyds.MassFunction({output_class : belief 
                        for output_class, belief in  zip(self.models[j].outputs, belief_entries)})

                # add to bpa list
                bpas.append(bpa)

            # now we perform the DS combination
            result_entry = bpas[0].combine_conjunctive(bpas[1:])

            # add it to cumulative list
            results.append(result_entry)
        
        return results

