"""
This class will serve as a general wrapper for a model
to be ensembled, providing an easier interface to ensemble
"""

from pydantic.typing import List, Union, Tuple
from numpy import ndarray
import numpy as np

class Model:

    def __init__(self, trained_model, model_type:str, output_classes:List[str],
                preprocess_function=None):
        '''
        Sets the model, its class type, and the number/which of outputs
        '''
        self.model = trained_model
        self.type = model_type.lower()
        self.outputs = output_classes
        self.preprocess_function = preprocess_function
        self.belief = None 

    def predict(self, pred_data: ndarray):
        '''
        This will handle the interpretation and prediction for a 
        particular model type 
        '''

        # preprocess if specified
        if self.preprocess_function:
            pred_data = self.preprocess_function(pred_data)

        # predict based on model type
        preds = None
        if self.type == 'sklearn':
            preds = self.model.predict(pred_data)

        return preds

    def setup_beliefs(self, evaluation_set: ndarray):
        '''
        Setup the model's belief values based on the evaluation set
        '''

        # get the predictions on the eval set
        preds = self.predict(evaluation_set[0])

        # group by the predicted classes and get belief for each
        # class based on pred
        counts = np.zeros([len(self.outputs), len(self.outputs)], dtype=np.float)
        for pred, truth in zip(preds, evaluation_set[1]):
            counts[pred, truth] += 1


        # ensure small delta to never have 0 or 1 beliefs
        counts = np.where(counts==0, 0.1, counts)

        # normalize and set to be beliefs
        # accounting for model accuracy of each class
        # and beliefs of classes given pred
        normed_1 = counts/counts.sum(axis=-1, keepdims=True)
        normed_2 = counts/counts.sum(axis=0, keepdims=True)
        normed = np.multiply(normed_1, normed_2)

        self.belief = normed