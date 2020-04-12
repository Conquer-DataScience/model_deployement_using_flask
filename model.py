# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:28:16 2020

@author: Meenakshi Lakshmanan
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Upload the IRIS Dataset
data = pd.read_csv('iris.csv')

# Now, lets prepare the training dataset
# X feature values, all columns except the last column
X = data.iloc[:,:-1] 
# Y - targe values, last column of dataframe
y = data.iloc[:,-1]

# Let's now split the data into 80% training set and 20% test set
x_train, x_test, y_train, y_test =  train_test_split(X,y, test_size=0.2, 
                                                     random_state=42)

# Train the model
model = LogisticRegression()
model.fit(x_train, y_train)
pickle.dump(model,open('model.pkl', 'wb'))
predictions = model.predict(x_test)

print(classification_report(y_test, predictions))
print("Accuracy Score:", accuracy_score(y_test, predictions))

