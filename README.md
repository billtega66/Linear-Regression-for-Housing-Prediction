# Linear-Regression-for-Housing-Prediction

Citation for Polynomial Regression code:
Author: Regenerative Today
Title: Polynomial Regression in Python - sklearn
Link: https://www.youtube.com/watch?v=nqNdBlA-j4w&list=PLbPfgyhPxomhPN6tOAsRTcTQ4kuhd3MO0&index=7

BIB:
RegenerativeToday. 2023. Polynomial Regression in Python
- sklearn. https://www.youtube.com/watch?v=nqNdBlA-
j4w.

Main contributions outside of following tutorial:

*Sorting Ocean Proximity Values

*Random State from "Split Data Into Training and Testing" is 42 and our test size is 0.3

*Tested the Polynomial degress 2, 3, 4 instead of 6 and 3 like the tutorial

*Used California Housing Dataset

*Imported: 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
