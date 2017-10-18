# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 11:11:28 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:37:00 2017

@author: Administrator
"""
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor

input='./allAndFeature.csv'
data= pd.read_csv(input)
angle=pd.DataFrame(data['angle'])
y=pd.DataFrame(data['y'])
feature=data.iloc[:,6:4103]#提取所有属性值
#feature['angle']=angle
feature['y']=y
#label=data[['x','height','width']]
label=data['angle']

X_train,X_test,Y_train,Y_test=train_test_split(feature,label,test_size=0.25,random_state=33)

def try_different_method(clf):
    clf.fit(X_train,Y_train)
    score = clf.score(X_test, Y_test)
    result = clf.predict(X_test)
    print('R-squared:', r2_score(Y_test, result))
    print('mean squared error is', mean_squared_error(Y_test, result))
    print('mean absolute error is', mean_absolute_error(Y_test, result))
    joblib.dump(clf, "train_model-y:angle.m", compress=3)
    '''
    plt.figure()
    plt.plot(np.arange(len(result)), Y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('score: %f'%score)
    plt.legend()
    plt.show()
    '''
#clf = DecisionTreeRegressor()
rfr=RandomForestRegressor()
etr=ExtraTreesRegressor()
gbr=GradientBoostingRegressor()
knn = neighbors.KNeighborsRegressor(weights='distance')

try_different_method(rfr)
try_different_method(etr)
try_different_method(gbr)
try_different_method(knn)
