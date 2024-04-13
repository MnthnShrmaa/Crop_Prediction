#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
data = pd.read_csv("Crop_recommendation.csv")
data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.duplicated().sum()


# In[8]:


data.describe()


# In[9]:


#data['label'] = data['label'].astype('category').cat.codes
x = data.drop('label', axis =1, inplace = False)
cor = x.corr()
cor


# In[10]:


import seaborn as sb
sb.heatmap(cor,annot = True, cbar = True, cmap = 'coolwarm')


# In[11]:


data['label'].value_counts()


# In[12]:


import matplotlib.pyplot as plt
sb.displot(data['P'])
plt.show()
sb.displot(data['N'])
# sb.histplot(data['N'])
plt.show()


# In[13]:


crop_dic = {'rice':1,
    'maize':2,
    'jute':3,
    'cotton':4,
    'coconut':5,
    'papaya':6,
    'orange':7,
    'apple':8,
    'muskmelon':9,
    'watermelon':10,
    'grapes':11,
    'mango':12,
    'banana':13,
    'pomegranate':14,
    'lentil':15,
    'blackgram':16,
    'mungbean':17,
    'mothbeans':18,
    'pigeonpeas':19,
    'kidneybeans':20,
    'chickpea':21,
    'coffee':22}

data['label_num'] = data['label'].map(crop_dic)
data['label_num'].value_counts()
# data['label_num'] = data['label'].astype('category').cat.codes
data.drop('label', axis = 1, inplace = True)
data.head()


# In[14]:


X = data.drop('label_num', axis = 1)
y = data['label_num']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[15]:


X


# In[16]:


X_train.shape


# In[17]:


X_test.shape


# In[18]:


from sklearn.preprocessing import MinMaxScaler 
ms = MinMaxScaler()

X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)
X_train


# In[19]:


from sklearn.preprocessing import StandardScaler 
s = StandardScaler()

X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)
X_train


# In[20]:


# from sklearn.preprocessing import RobustScaler 
# rs = RobustScaler()

# rs.fit(X_train)
# X_train_rs = rs.transform(X_train)
# X_test_rs = rs.transform(X_test)
# X_train_rs


# In[21]:


# from sklearn.preprocessing import Normalizer 
# n = Normalizer()

# n.fit(X_train)
# X_train_n = n.transform(X_train)
# X_test_n = n.transform(X_test)
# X_train_n


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreeClassifier(),
}

for name, algo in models.items():
    algo.fit(X_train,y_train)
    ypred = algo.predict(X_test)
    
    print(f"{name}  with accuracy : {accuracy_score(y_test,ypred)}")


# In[23]:


rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
ypred = rfc.predict(X_test)
a = accuracy_score(y_test, ypred) 
print(f' accuracy score : {a}\n accuracy percentage :{a*100}%')


# In[24]:


import numpy as np
def prediction(N, P, K, temp, hum, ph, rainfall):
    feature_arr = np.array([[N, P, K, temp, hum, ph, rainfall]])
    feat_tran = ms.fit_transform(feature_arr)
    feat_tran = s.fit_transform(feat_tran)
    new_pred = rfc.predict(feat_tran).reshape(1,-1)
    return new_pred[0]
    


# In[25]:


N = float(input('Enter the amount of Nitrogen present in soil:'))
P = float(input('Enter the amount of Phosphorous present in soil:'))
K = float(input('Enter the amount of Potassium present in soil:'))
temp = float(input('Enter the Temperature of the area:'))
hum = float(input('Enter the amount of Humidity present in environment:'))
ph = float(input('Enter the ph of soil:'))
rainfall = float(input('Enter the amount of rainfall in the area:'))

pred = prediction(N,P, K, temp, hum, ph, rainfall)

crops = {1:'rice', 
         2:'maize', 
         3:'jute', 
         4:'cotton', 
         5:'coconut', 
         6:'papaya', 
         7:'orange', 
         8:'apple', 
         9:'muskmelon', 
         10:'watermelon', 
         11:'grapes', 
         12:'mango', 
         13:'banana', 
         14:'pomegranate', 
         15:'lentil', 
         16:'blackgram', 
         17:'mungbean', 
         18:'mothbeans', 
         19:'pigeonpeas', 
         20:'kidneybeans', 
         21:'chickpea', 
         22:'coffee'} 

if pred[0] in crops: 
    crop = crops[pred[0]] 
    print(f' The prediction is that the {crop} crop would be suitable for this area') 
else: 
    print('Environment not suitable for any crop')


# In[ ]:




