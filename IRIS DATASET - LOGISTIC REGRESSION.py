#!/usr/bin/env python
# coding: utf-8

# IRIS DATASET 

# It is a Typical Machine Learning Classification Problem.The iris flower is found in three species. We must determine which type a newly supplied flower belongs to. All three species' samples are displayed in the following figure.

# In[49]:


# import image module 
from IPython.display import Image 
Image(url="Species_IRIS.png", width=600, height=600) 


# In[50]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
sns.set(style="white", color_codes=True)


# In[51]:


iris=pd.read_csv("Iris.csv")
iris.head()


# In[52]:


iris.tail()


# In[53]:


# To count the number of unique entities in a species.
iris["Species"].value_counts() 


# Scatter Plot : To visualize how the data looks like...

# In[18]:


sns.FacetGrid(iris,hue="Species",height=6).map(plt.scatter,"PetalLengthCm","SepalWidthCm").add_legend()


# LOGISTIC REGRESSION

# In[54]:


# Converting Categorical Variables to Numerical...
mapping = {'Setosa': 0,'Versicolor': 1,'virginica':2}
iris["Species"] = iris["Species"].map(mapping)


# In[55]:


print("NaN values before mapping:", iris["Species"].isna().sum())


# In[56]:


iris.head()


# In[57]:


iris.tail()


# In[62]:


# Preparing inputs and outputs...
X=iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
y=iris[['Species']].values 


# In[65]:


# Importing ML Algorithm
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)


# In[66]:


# Accuracy of Model
model.score(X,y)


# In[68]:


# Prediction
expected = y
predicted = model.predict(X)
predicted


# In[70]:


# Model Fitness
from sklearn import metrics
print(metrics.classification_report(expected, predicted))


# In[71]:


print(metrics.confusion_matrix(expected,predicted))

