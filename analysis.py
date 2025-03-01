#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("dataset.csv")


# In[37]:


plt.figure(figsize=(10, 5))
plt.plot(df["price"], df["area"], marker="o", linestyle="-", color="b", label="Value")
plt.xlabel("price")
plt.ylabel("area")
plt.title("Price vs area")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.show()


# In[38]:


df.info()


# In[39]:


df["furnishingstatus"].unique()


# In[40]:


replacements = {
    "yes":1,
    "no":0,
    "furnished":1,
    "semi-furnished":0.5,
    "unfurnished":0
}
df.replace(replacements,inplace=True)
df.info()


# In[41]:


df.corr()


# In[42]:


corr_mat = df.corr()
plt.figure(figsize=(8,6))
sb.heatmap(corr_mat,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
plt.title("correlation heat map")
plt.show()


# In[84]:


params = corr_mat[(corr_mat['price']>0.5) & (corr_mat['price']<1)]['price']
params


# In[85]:


features = [k for k in params.index]
features


# In[86]:


X = df[features]
y = df['price']
X,y


# In[87]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=34)


# In[88]:


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[89]:


# for pred, actual in zip(y_pred, y_test):
#     print(pred, "--", actual," --diff--", pred-actual)


# In[90]:


mean_squared_error(y_test, y_pred)


# In[91]:


r2_score(y_test, y_pred)


# In[ ]:




