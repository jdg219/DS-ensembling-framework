"""
This file will generate a dataset where there are points randomly
distributed in each quadrant of the euclidean plane (ensured to be away from the x
and y axis) and have corresponding class labels of the quadrant they are in
minus 1 (so we have classes 0-3 instead of 1-4) where the 0th quadrant is 
the positive x and y locations, with ascending quadrant going counterclockwise

Image:

        |
    1   |   0
        |
----------------
        |
    2   |   3
        |

"""


import random
import os
import numpy as np
import pandas as pd

# create arrays
x_points = np.zeros(400, dtype=float)
y_points = np.zeros(400, dtype=float)
labels = -1 * np.ones(400, dtype=int)

# generate first 100 points
x_points[:100] = 1 + 25*np.random.rand(100)
y_points[:100] = 1 + 25*np.random.rand(100)
labels[:100] = 0

# generate second 100 points
x_points[100:200] = -1 + -25*np.random.rand(100)
y_points[100:200] = 1 + 25*np.random.rand(100)
labels[100:200] = 1

# generate third 100 points
x_points[200:300] = -1 + -25*np.random.rand(100)
y_points[200:300] = -1 + -25*np.random.rand(100)
labels[200:300] = 2

# generate final 100 points
x_points[300:] = 1 + 25*np.random.rand(100)
y_points[300:] = -1 + -25*np.random.rand(100)
labels[300:] = 3

# construct output 
df = pd.DataFrame({'x_points': x_points, 'y_points': y_points, 'labels': labels})
df.to_csv('quadrant_dataset.csv')
