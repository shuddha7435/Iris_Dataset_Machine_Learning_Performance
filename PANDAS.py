#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import datasets


# In[3]:


iris_dataset = datasets.load_iris()


# In[4]:


X = iris_dataset.data[:,:2]


# In[5]:


x_count = len(X.flat)
x_min = X[:, 0].min() - .5
x_max = X[:, 0].min() + .5
x_mean = X[:, 0].mean()


# In[6]:


x_count, x_min, x_max, x_mean


# In[7]:


import sys
print ('Python: {}'.format(sys.version))


# In[8]:


import sys
print ('Python: {}'.format(sys.version))
import scipy
print ('scipy: {}'.format(scipy.__version__))
import numpy
print ('numpy: {}'.format(numpy.__version__))
import matplotlib
print ('matplotlib: {}'.format(matplotlib.__version__))
import pandas
print ('pandas: {}'.format(pandas.__version__))
import sklearn
print ('sklearn: {}'.format(sklearn.__version__))


# In[9]:


import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[10]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width','petal-length','petal-width','class']
dataset = pandas.read_csv(url, names = names)


# In[11]:


print(dataset.shape)


# In[12]:


print(dataset.head(30))


# In[13]:


print(dataset.describe())


# In[14]:


print(dataset.groupby('class').size())


# In[20]:


dataset.plot(kind='box', subplots = True, layout=(2,2), sharex = False,sharey = False )
plt.show()


# In[21]:


dataset.hist()
plt.show()


# In[22]:


scatter_matrix(dataset)
plt.show()


# In[24]:


array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)


# In[25]:


seed = 6
scoring = 'accuracy'


# In[29]:


#Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name,model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold,scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[ ]:




