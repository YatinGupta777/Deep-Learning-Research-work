from __future__ import print_function
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from dbn.tensorflow import SupervisedDBNClassification

df_train = pd.read_csv('example.csv')

# CHecking whether data is imbalanced
# I is output variable
target_count = df_train.I.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

#target_count.plot(kind='bar', title='Count (target)');
'''Random under-sampling'''
# Class count
count_class_0, count_class_1 = df_train.I.value_counts()

# Divide by class
df_class_0 = df_train.loc[df_train['I'] == 0]
df_class_1 = df_train.loc[df_train['I'] == 1]

df_class_0_under = df_class_0.sample(count_class_1)
df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)

df_train_under = df_train_under.reset_index(drop=True)

print('Random under-sampling:')
print(df_train_under.I.value_counts())
'''Source'''
#https://github.com/albertbup/deep-belief-network

#df_train_under.I.value_counts().plot(kind='bar', title='Count (target)'); 
#digits = df_train_under
X, Y = df_train_under.loc[:, df_train_under.columns != 'I'], df_train_under.I

# Data scaling
X = (X / 16).astype(np.float32)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)

'''Converting to numpy array'''
X_train = X_train.values
Y_train = Y_train.values
classifier.fit(X_train, Y_train)

# Save the model
classifier.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred)) 
 
 

#https://github.com/echen/restricted-boltzmann-machines/blob/master/rbm.py
  
#Explanation
#http://deeplearning.net/tutorial/rbm.html  
  
#Datset
#https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

