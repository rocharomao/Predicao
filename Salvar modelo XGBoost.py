#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Train XGBoost model, save to file using joblib, load and make predictions
from numpy import loadtxt
from xgboost import XGBClassifier
from joblib import dump
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt(r"C:\Users\glauber.romao\Documents\Glauber\Pessoal\diabetes.csv", delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
# save model to file
dump(model, "nomeModelo.dat")
#print("Saved model to: pima.joblib.dat")
 


# In[5]:


# some time later...
 
# load model from file
loaded_model = load("nomeModelo.dat")

# make predictions for test data
predictions = loaded_model.predict(X_test)
print(predictions)
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[24]:


#Exemplo de utilização do modelo Tem ou não tem a doença
# inModelTrue = [1., 181.,84.,21., 193.,35.9,0.586,51.]
# inModelFalse = [1.  , 90.  , 62.  , 12.  , 43.  , 27.2 ,  0.58, 24. ]
# loaded_model.predict(inModelFalse)


# In[ ]:




