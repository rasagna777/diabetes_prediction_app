#!/usr/bin/env python
# coding: utf-8

# In[75]:


pip install joblib


# In[76]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib


# In[77]:


df = pd.read_csv("diabetes.csv")
df


# In[78]:


X = df.drop(columns=["class"])
y = df["class"]


# In[79]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[80]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = DecisionTreeClassifier(random_state=42)
scores = cross_val_score(model, X_scaled, y, cv=kf)
mean_accuracy = accuracy_scores.mean()
print("Mean Accuracy:", mean_accuracy)
print("Cross-validation scores:", scores)


# In[81]:


#Train final model
model.fit(X_scaled, y)

#Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# In[ ]:




