# Reasoning Under Uncertainty Project

Jared Gallina
jgallin2@jh.edu


## Objective

The goal of this software package is to provide a broadly applicable ML/DL ensembling
library that leverages Dempster-Schafer Theory for ensembling, specifically Dempster
Combination.


## Brief Description

#### Installation

The package is able to be installed directly in a python environment by simply running:

```
pip install -e
```

Then the necessary dependencies can be installed with the requirements.txt file via:

```
pip install -r requirements.txt
```

both commands are run from the root of the repository.

#### Functionality

The package operates off of the ds_ensemble/DSEnsemble file as the backbone for all combination.
Usage of the class for combination is demonstrated in the examples subfolder. Additionally,
the testing folder holds the python files used for evaluating DS ensembling versus classic
ensembing.


## Limitations

The current skeleton is only able to combine sklearn models predicting all classes. 
Deep Learning models will be added as well as the ability to combine models that inference
on different subsets of outputs.


## Roadmap

A brief description of accomplished tasks and those yet to be completed.

#### Currently Implemented

- Basic sklearn ensemble example
- Skeleton of class to handle ensemble using Dempster Combination
- Package setup to install/manage dependencies
- Combination class can handle sklearn models
- Basic evaluation of DS ensembling versus majority vote ensembling

#### Work in Progress

- Expand combination class to handle tensorflow and pytorch models
- Create more diverse model ensembling example
- Expand combination class to handle different levels of prediction {a vs b vs c} or {a&b vs c}
- Create tests for traditional vs DS ensemble
    * Diverse model architectures
    * Crossval fold trained models
    * Different levels of prediction benefit evaluation

#### Potential Extended Work

- Implement t-norm DS Ensembling
- Implement auto grouping of outputs to have best performing info for a model
    * For example, auto group model outputs to predict the most pure/accurate results
     