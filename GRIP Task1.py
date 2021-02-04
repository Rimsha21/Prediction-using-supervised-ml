
# coding: utf-8

# ## Author-  Rimsha Virmani
# ## GRIP @ The Sparks Foundation
# 
# ## Task-1 : Predict the percentage of a student based on the no. of study hours  studied using Supervised Machine Learning
# 
# ## Problem statement: What will be predicted score if a student studies for 9.25 hrs/day?
# 
# ## Solution:  For predicting the student's score based on the number of hours' studied, I have used Linear Regression 
# 
# ## Dataset  : https://bit.ly/3kXTdox

# In[25]:


# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ## 1. Reading and loading the data from url

# In[3]:


url ="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data= pd.read_csv(url)
print("Data loaded successfully!")
data.head()


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


data.shape


# In[24]:


#checking for null values
data.isnull().sum()


# ## 2. Performing Data Visualization

# In[7]:


#Plot of Scores Distribution
data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours Vs Percentage Score')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.show()


# # By seeing the graph, it can be assumed that there is a positive relationship between the number of hours studied and the percentage score

# ## 3. Data Preprocessing

# In[8]:


# Data is divided into "attributes" and "labels"
X = data.iloc[:, :-1].values
y= data.iloc[:, 1].values


# ## 4. Model training

# In[12]:


#Splitting of data into training and testing sets
X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
regressor= LinearRegression()
regressor.fit(X_train.reshape(-1,1), y_train)
print("Training Completed")


# ## Plotting the Line of Regression

# In[15]:


#Visualizing the best fit line of Regression.
line= regressor.coef_*X + regressor.intercept_
plt.scatter(X,y)
plt.plot(X,line, color='black')
plt.show()


# ## Predicting the data

# In[16]:


print(X_test) #Testing data
y_pred= regressor.predict(X_test) #Model predictions


# ## 7. Comparision of Actual result vs Prediction result

# In[17]:


df = pd.DataFrame({'Actual Result': y_test, 'Predicted Result': y_pred})  
df


# ## 8. Estimating training and test score

# In[18]:


# Training score
print("Training Score: ", regressor.score(X_train,y_train))
# Test Score
print("Test Score: ", regressor.score(X_test,y_test))


# In[21]:


# Visualizing the difference between actual and predicted result
df.plot(kind='bar', figsize=(5,5))
plt.grid(linewidth='0.5', color='blue')
plt.grid(linewidth='0.5', color='red')


# In[22]:


# Testing with own Data
Hours = 9.25
test= np.array([Hours]).reshape(-1,1)
prediction = regressor.predict(test)
print("Number Of Hours = {}".format(Hours))
print("Predicted Score= {}".format(prediction[0]))


# ## 9. Model Evaluation

# In[23]:


# Let's calculate different errors to compare the model performance and predict accuracy.
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
print("Mean Absolute Error: ",mean_absolute_error(y_test,y_pred))
print("Mean Squared Error: ",mean_squared_error(y_test,y_pred))
print("R2 score: ", r2_score(y_test,y_pred))


# ## Conclusion

# # The prediction using supervised ML has  been performed successfully.
