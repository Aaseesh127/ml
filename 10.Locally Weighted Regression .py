#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
tou=0.80
X_train=np.array(list(range(3,33))+[3.2,4.2])
X_train=X_train[:,np.newaxis]
print(X_train)
Y_train=np.array([1,2,1,2,1,1,3,4,5,4,5,6,5,6,7,8,9,10,11,11,12,11,11,12,13,16,17,19,18,34,21,22])
X_test=np.array([i/10. for i in range(400)])
X_test=X_test[:,np.newaxis]
Y_test=[]
count=0
for r in range(len(X_test)):
    try:
        wt=np.exp(-np.sum((X_train-X_test[r])**2,axis=1)/(2*tou)**2)
        w=np.diag(wt)
        fact1=np.linalg.inv(X_train.T.dot(w).dot(X_train))
        parameter=fact1.dot(X_train.T).dot(w).dot(Y_train)
        prediction=X_test[r].dot(parameter)
        Y_test.append(prediction)
        count=count+1
    except:
        pass
Y_test=np.array(Y_test)
plt.plot(X_train.squeeze(),Y_train,'o')
plt.plot(X_test.squeeze(),Y_test,'*')
plt.show()
Y_test


# In[ ]:




