"""
This class will serve as a general wrapper for a model
to be ensembled, providing an easier interface to ensemble
"""

from pydantic.typing import List, Union, Tuple
from numpy import ndarray

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

    