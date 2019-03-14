#!/usr/bin/env python
# coding: utf-8

# # Case Study Fintech Company
# 
# - The company wants to give paid subscripton to its customers which will allow to track their finances in one place.
# - To attract customers the company has released a free version of the app with some main features unlocked.
# - Our task is to indentify customers who will NOT enroll in the paid products so additional offers can be made to them
# 
# ## DATA
# 
# - We have access to the data of customers behaviour with the app. 
# - This data include Date-Time installation of the application, features that customer engaged within the app. 
# - The App behaviour is charecterized as the list of app screens customer looked at and whether they engaged in the min-financial games available within the app
# - The company allows only a 24 hour trial to the customer and then provides the user with offers for the premium version of the app.
# 
# ## Dataset
# - Our Dataset consists of :
#     - user : The ID of the user
#     - first_open : When the app was first opened
#     - dayofweek : The day of the week 0 being Sunday and 6 being Satuday
#     - age : Age of the user
#     - screen_list : list of screens accesed by the user
#     - numscreens : Number of screens the user has seen
#     - minigame : Whether the minigame is played by the user
#     - liked : Different pages of the app contains like feature and how many times the user used it.
#     - used_premium_feature : A user has used the premium feature or not.
#     - enrolled : User enrolled to the paid product.
#     - enrolled_date : Date and Time when user enrolled to the paid product
#     
# ### Importing Required Iibraries to start the EDA process    


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from dateutil import parser



dataset = pd.read_csv('appdata10.csv')



dataset.describe()


# Coverting our hour column in dataset to an int type


dataset['hour'] = dataset.hour.str.slice(1,3).astype(int)


# Cerating a temporary dataset with columns we don't want


dataset2 = dataset.copy().drop(columns = ['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])



dataset2.head()


# Plotting histograms to get better insights of our dataset


plt.suptitle('Histograms of numerical columns', fontsize = 12)
for i in range (1, dataset2.shape[1] + 1):
    plt.subplot(3,3,i)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i-1])
    vals = np.size(dataset2.iloc[:, i -1 ].unique())
    plt.hist(dataset2.iloc[:, i- 1], bins = vals, color = "#3F5D7D")
    plt.subplots_adjust(hspace= 0.9, wspace= 0.9)


# - Now the we have plotted our dataset we'll convert out insights into correlation by using correlation plots
# - We do this in order to find out which of our attributes are important and at what magnitude they affect out final outcome i.e the users enrolled for premium version 


dataset2.corrwith(dataset.enrolled).plot.bar(figsize = (20,10),
                                             title = 'Correlation with Response Variable',
                                             fontsize = 15, rot = 45,
                                             grid = True)


# ### Making vague conslusions from our correlation plot
# - dayofweek is positively related i.e later the day of the week more likely to enroll
# - hour is neagtively related i.e earlier in the day more likely to enroll
# - age is negatively related i.e more young the user is more likely to enroll
# - numscreens is positvely related i.e more screens visited more likely to enroll
# - minigame is positvely related i.e user who played minigame is more likely to enroll
# - used_premium_feature is negatively related i.e user who used premium feature is unlikely to enroll
# 
# ### Now before Model Building We'll build a correlation matrix


sn.set(style='white', font_scale=2)

# Compute the correlation matrix
corr = dataset2.corr()

# Generate the mask for upper traingle
mask = np.zeros_like(corr,dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Setup the matplotlib Figure
f ,ax = plt.subplots(figsize = (22,9))
f.suptitle('Correlation Matrix', fontsize = 40)

# Generate a Custome dividing colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

#Draw heatmap with mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3 , center=0, square=True, linewidths=0.5, cbar_kws={"shrink" : 0.5})


# ### Now we'll format the listed screens 
# - We have a csv that gives us the list of screens that are used more frequently
# - We'll format the screen_list using this csv file



top_screens = pd.read_csv('top_screens.csv').top_screens.values




top_screens




dataset['screen_list'] = dataset.screen_list.astype(str) + ','




for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset['screen_list'] = dataset.screen_list.str.replace(sc+",","")
    




dataset['Other'] = dataset.screen_list.str.count(",")
dataset.drop(columns=['screen_list'])


# - We created a sperate column for each top screen that was accessed by the user 
# - We counted all the other screens visited in a single attribute 'Other'
# - Then dropped the screen_list attruibute 
# 
# ### Funnels
# - Now we'll be creating the funnels for every screens that are from the same module
# - For eg : There are screen for Savings, Loan, Credit and CC



saving_screens = ["Saving1","Saving2","Saving2Amount","Saving4","Saving5","Saving6","Saving7","Saving8","Saving9","Saving10"]
dataset['SavingsCount'] = dataset[saving_screens].sum(axis = 1)
dataset.drop(columns= saving_screens)

cm_screens = ["Credit1","Credit2","Credit3","Credit3Container","Credit3Dashboard"]
dataset['CMCount'] = dataset[cm_screens].sum(axis = 1)
dataset.drop(columns= cm_screens)

cc_screens = ["CC1","CC1Category","CC3"]
dataset['CCCount'] = dataset[cc_screens].sum(axis = 1)
dataset.drop(columns= cc_screens)

loan_screens = ["Loan","Loan2","Loan3","Loan4"]
dataset['LoansCmount'] = dataset[loan_screens].sum(axis = 1)
dataset.drop(columns= loan_screens)




dataset.to_csv('new_appdata10.csv', index = False)


# ## Now we'll be building our model



#### Importing Libraries ####

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import time

app_dataset = pd.read_csv('new_appdata10.csv')


#### Data Pre-Processing ####

# Splitting Independent and Response Variables
response = app_dataset['enrolled']
app_dataset = app_dataset.drop(columns=['enrolled','screen_list', 'enrolled_date', 'first_open'])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(app_dataset, response,
                                                    test_size = 0.2,
                                                    random_state = 0)




# Removing Identifiers
train_identity = X_train['user']
X_train = X_train.drop(columns = ['user'])
test_identity = X_test['user']
X_test = X_test.drop(columns = ['user'])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2




#Model Building

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0 ,penalty='l2',solver = 'lbfgs')
# We are using l1 penalty here beacuse let's suppose there is a screen that happens just before the enrollment screen
# Thus it will have a bigger coefficent in our Logistic Regression Model so l1 penalty will be higher on the screen 
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)




from sklearn.metrics import confusion_matrix, precision_recall_fscore_support,classification_report

cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))




# Plotting our Seaborn Hetmap
df_cm = pd.DataFrame(cm, index=(0,1), columns=(0,1))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')




# Using K_fold Cross Validation to check for overfitting
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y= y_train, cv =10)
print("Logistic Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))





