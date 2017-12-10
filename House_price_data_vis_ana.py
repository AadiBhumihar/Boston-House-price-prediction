
# coding: utf-8

# In[1]:


import pandas as pd
test_df = pd.read_csv('train.csv')
test_df.head()


# In[2]:


import pylab as plt
plt.style.use(style='ggplot')
plt.plot(test_df.LotFrontage,test_df.SalePrice,'ro')
plt.show()


# In[5]:


import numpy as np
import pylab as plt
group_by_price = pd.cut(test_df.LotFrontage, np.arange(0, 350, 35))
price_grouping = test_df.groupby(group_by_price).mean()
price_grouping['SalePrice'].plot.bar()
plt.show()


# In[6]:


import numpy as np
import pylab as plt
group_by_price = pd.cut(test_df.LotFrontage, np.arange(0, 350, 35))
price_grouping = test_df.groupby(group_by_price).mean()
price_grouping['SalePrice'].plot.bar()
plt.show()


# In[7]:


zone_grouping = test_df.groupby('MSZoning').mean()
zone_grouping['SalePrice'].plot.bar()
plt.show()


# In[8]:


class_grouping = test_df.groupby('MSSubClass').mean()
class_grouping['SalePrice'].plot.bar()
plt.show()


# In[9]:


class_grouping = test_df.groupby(['MSSubClass','MSZoning']).mean()
class_grouping


# In[10]:


class_grouping = test_df.groupby(['MSSubClass','MSZoning']).mean()
class_grouping['SalePrice'].plot.bar()
plt.show()


# In[11]:


test_df.head()


# In[13]:


test_df.groupby(['PoolArea','PoolQC']).head(80)


# In[15]:


test_df.groupby(['PoolArea','PoolQC']).mean()


# In[16]:


pool_class = test_df.groupby(['PoolArea','PoolQC'],as_index=False).mean()
pool_class['SalePrice'].plot.bar()
plt.show()


# In[17]:


pool_class = test_df.groupby('Alley',as_index=False).mean()
pool_class['SalePrice'].plot.bar()
plt.show()


# In[18]:


pool_class = test_df.groupby('Alley',as_index=False).mean()


# In[19]:


test_df.groupby('Alley',as_index=False).mean()


# In[20]:


test_df.groupby('Alley',as_index=False).count()


# In[ ]:


test_df.groupby('Alley',as_index=False).count()


# In[21]:


pool_class = test_df.groupby('YearBuilt',as_index=False).mean()
pool_class['SalePrice'].plot.bar()
plt.show()


# In[22]:


test_df.YearBuilt.min(axis=0)


# In[23]:


test_df.YearBuilt.max(axis=0)


# In[24]:


year_group = pd.cut(test_df.YearBuilt, np.arange(1870, 2010, 14))
year_class = test_df.groupby(year_group,as_index=False).mean()
year_class['SalePrice'].plot.bar()
plt.show()


# In[25]:


year_group = pd.cut(test_df.YearBuilt, np.arange(1870, 2010, 14))
year_class = test_df.groupby(year_group,as_index=False).mean()


# In[26]:


year_class


# In[28]:


year_group = pd.cut(test_df.YearBuilt, np.arange(1870, 2010, 14))
test_df.groupby(test_df.YearBuilt,as_index=False).mean()


# In[29]:


year_group = pd.cut(test_df.YearBuilt, np.arange(1870, 2010, 14))
test_df.groupby(test_df.YearBuilt,as_index=False).mean()


# In[30]:


year_group = pd.cut(test_df.YearBuilt, np.arange(1870, 2010, 10))
year_group


# In[31]:


year_group = pd.cut(test_df.YearBuilt, np.arange(1870, 2010, 10))
year_class = test_df.groupby(year_group,as_index=False).mean()
year_class['SalePrice'].plot.bar()
plt.show()


# In[32]:


import pandas as pd
test_df = pd.read_csv('train.csv')
test_df.head()


# In[34]:


import pandas as pd
test_df = pd.read_csv('train.csv')
fence_class = test_df.groupby(test_df['Fence'],as_index=False).mean()
year_class['SalePrice'].plot.bar()
plt.show()


# In[35]:


import pandas as pd
test_df = pd.read_csv('train.csv')
fence_class = test_df.groupby(test_df['Fence'],as_index=False).mean()
fence_class['SalePrice'].plot.bar()
plt.show()


# In[7]:


import pandas as pd
test_df = pd.read_csv('train.csv')
fence_class = test_df.groupby(test_df['Fence'],as_index=False).mean()
fence_class['SalePrice'].plot.bar()
plt.show()


# In[8]:


import pandas as pd
test_df = pd.read_csv('train.csv')
fence_class = test_df.groupby(test_df['Fence'],as_index=False).mean()
fence_class


# In[10]:


s = pd.Series(list(test_df['Fence']))
pd.get_dummies(s)


# In[11]:


s = pd.Series(list(test_df['Fence'])).rename(columns=lambda x: 'Fence_' + str(x))
pd.get_dummies(s)


# In[12]:


pd.get_dummies(test_df['Fence']).rename(columns=lambda x: 'Fence_' + str(x))


# In[20]:


test_df = pd.read_csv('train.csv')
dummies = pd.get_dummies(test_df['Fence']).rename(columns=lambda x: 'Fence_' + str(x))
pd.concat([test_df, dummies], axis=1)
test_df.drop(['Fence'], inplace=True, axis=1)


# In[21]:


test_df = pd.read_csv('train.csv')
dummies = pd.get_dummies(test_df['Fence']).rename(columns=lambda x: 'Fence_' + str(x))
pd.concat([test_df, dummies], axis=1)


# In[22]:


test_df = pd.read_csv('train.csv')
dummies = pd.get_dummies(test_df['Fence']).rename(columns=lambda x: 'Fence_' + str(x))
pd.concat([test_df, dummies], axis=1)
test_df.drop(['Fence'],  axis=1)


# In[23]:


test_df = pd.read_csv('train.csv')
dummies = pd.get_dummies(test_df['Fence']).rename(columns=lambda x: 'Fence_' + str(x))
test_df = pd.concat([test_df, dummies], axis=1)
test_df =test_df.drop(['Fence'],  axis=1)


# In[24]:


test_df


# In[25]:


df =test_df['MSZoning']


# In[26]:


df


# In[29]:


df.name


# In[31]:


df.unique()


# In[34]:


df.dtypes


# In[1]:


import pandas as pd
test_df = pd.read_csv('train.csv')
test_df.head()


# In[2]:


import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt


# In[3]:


test_df.head()


# In[4]:


test_val = test_df.values


# In[5]:


np.shape(test_val)


# In[7]:


test_df.convert_objects(convert_numeric=True)


# In[8]:


train_df = pd.read_csv('train.csv')


# In[9]:


train_df.head()


# In[10]:


train_df.convert_objects(convert_numeric=True)


# In[11]:


train_df.dtypes


# In[16]:


train_val[:,80]


# In[17]:


y = train_val[:,80]/1000


# In[18]:


y


# In[19]:


x1 = train_val[:,4]/100


# In[20]:


x1


# In[21]:


plt.plot(x1,y)
plt.show()


# In[22]:


plt.scatter(x1,y)
plt.show()


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


print ("Train data shape:", train.shape)
print ("Test data shape:", test.shape)


# In[4]:


train.head()


# In[5]:


import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


# In[6]:


train.SalePrice.describe()


# In[7]:


print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()


# In[8]:


target = np.log(train.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()


# In[9]:


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes


# In[ ]:




