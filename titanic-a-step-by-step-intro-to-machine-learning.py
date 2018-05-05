# -*- coding: utf-8 -*-
"""
Created on Sat May  5 16:16:49 2018

@author: Administrator
"""

# practice follow https://www.kaggle.com/ydalat/titanic-a-step-by-step-intro-to-machine-learning

#Load libraries for analysis and visualization
import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

#%matplotlib inline
import plotly.offline as py

from matplotlib import pyplot
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter

# machine learning libraries
import xgboost as xgb

import seaborn as sns

import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier)

from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# 1.2 loading dataset
train = pd.read_csv('C:/Users/Administrator/Desktop/titalic/train.csv')
test = pd.read_csv('C:/Users/Administrator/Desktop/titalic/test.csv')

PassengerId = test['PassengerId']

train.head(5)
