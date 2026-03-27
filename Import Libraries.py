import numpy as np 
import pandas as pd
import seaborn as s
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
#optimum parameter choosing 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pickle
import os 
import warnings
warnings.filterwarnings('ignore')



Load Dataset  
# Data file is in the user’s folder 
os.chdir ('C:\\Noble\\Training\\Top Mentor\\Training\\Presentation\\Project\\Project to Explain Classifciation\\')
data = pd.read_csv('data.csv')
display (data)

Display  Shape 

display (data.shape)

Create Variable df from data 
df = data
display (df)

print distinct value count 
display (df['diagnosis'].value_counts())

Print Data Types 
print (df.dtypes)

Update Date Type to Category  - Diagnosis  

df['diagnosis'] = df['diagnosis']. astype('category')
print (df.dtypes)


Label Encoding – Diagnosis  

df['diagnosis'] = df['diagnosis'].cat.codes
print ('********')
print(df.dtypes)

Display the data set 
df.head()


Create X variable with out diagnosis column 
x= df.drop ('diagnosis',axis =1).drop('id',axis =1)
display (x)

Create Y variable (diagnosis)
y = df['diagnosis']
display (y)

Extract all column names 
col = x. columns
display (col)

Null Check 
display (x.isnull().sum())


Print Co relation 
co_rel= x.corr()
display (co_rel)

Heat map with co relation 
plt.rcParams['figure.figsize']=(20,12)
s.set(font_scale=1.4)
# In co relation 1 is the highest and -1 is lowest
s.heatmap (co_rel,cmap = 'coolwarm',annot = True)
plt.show()

Heat map with out corelation value – Annot = None 
plt.rcParams['figure.figsize']=(20,12)
s.set(font_scale=1.4)
# In co relation 1 is the highest and -1 is lowest
s.heatmap (co_rel,cmap = 'coolwarm',annot = None)
plt.show()

Create Box Plot 
#box plot to check the outliers. Not going to remove outliers since this data is important. 
# Observation, when 'diagnosis' is "B", the values are lower 
plt.rcParams['figure.figsize']=(20,8)
f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
s.boxplot ( x= df['diagnosis'], y = df['radius_mean'], ax = ax1)
s.boxplot (x= df['diagnosis'], y = df['texture_mean'], ax = ax2)
s.boxplot (x= df['diagnosis'], y = df['perimeter_mean'], ax = ax3)
s.boxplot (x= df['diagnosis'], y = df['area_mean'] , ax = ax4)
s.boxplot (x= df['diagnosis'], y = df['smoothness_mean']  , ax = ax5)
f .tight_layout()

f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
s.boxplot (x= df['diagnosis'], y = df['compactness_mean'], ax = ax1)
s.boxplot (x= df['diagnosis'], y = df['concavity_mean'] , ax = ax2)
s.boxplot (x= df['diagnosis'], y = df['concave points_mean'] , ax = ax3)
s.boxplot (x= df['diagnosis'], y = df['symmetry_mean'], ax = ax4)
s.boxplot (x= df['diagnosis'], y = df['fractal_dimension_mean'] , ax = ax5)
f .tight_layout()

 Distribution plot  
g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "radius_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, 'texture_mean', hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, 'perimeter_mean', hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "area_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "smoothness_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "compactness_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "concavity_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "concave points_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "symmetry_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "fractal_dimension_mean", hist = False, rug = True)
plt.show()

Box Plot 
plt.rcParams['figure.figsize']=(20,8)
f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
s.boxplot (x= df['diagnosis'], y = df['radius_se'], ax = ax1,palette = 'cubehelix')
s.boxplot (x= df['diagnosis'], y = df['texture_se'], ax = ax2,palette = 'cubehelix')
s.boxplot (x= df['diagnosis'], y = df['perimeter_se'] , ax = ax3,palette = 'cubehelix')
s.boxplot (x= df['diagnosis'], y = df['area_se'], ax = ax4,palette = 'cubehelix')
s.boxplot (x= df['diagnosis'], y = df['smoothness_se'], ax = ax5,palette = 'cubehelix')
f .tight_layout()

f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
s.boxplot (x= df['diagnosis'], y = df['compactness_se'], ax = ax1,palette = 'cubehelix')
s.boxplot (x= df['diagnosis'], y = df['concavity_se'], ax = ax2,palette = 'cubehelix')
s.boxplot (x= df['diagnosis'], y = df['concave points_se'],  ax = ax3,palette = 'cubehelix')
s.boxplot (x= df['diagnosis'], y = df['symmetry_se'], ax = ax4,palette = 'cubehelix')
s.boxplot (x= df['diagnosis'], y = df['fractal_dimension_se'], ax = ax5,palette = 'cubehelix')
f .tight_layout()

Box Plot 
plt.rcParams['figure.figsize']=(20,8)
f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
s.boxplot (x= df['diagnosis'], y = df['radius_worst'], ax = ax1,palette = 'coolwarm')
s.boxplot (x= df['diagnosis'], y = df['texture_worst'], ax = ax2,palette = 'coolwarm')
s.boxplot (x= df['diagnosis'], y = df['perimeter_worst'], ax = ax3,palette = 'coolwarm')
s.boxplot (x= df['diagnosis'], y = df['area_worst'], ax = ax4,palette = 'coolwarm')
s.boxplot (x= df['diagnosis'], y = df['smoothness_worst'], ax = ax5,palette = 'coolwarm')
f .tight_layout()

f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
s.boxplot (x= df['diagnosis'], y = df['compactness_worst'], ax = ax1,palette = 'coolwarm')
s.boxplot (x= df['diagnosis'], y = df['concavity_worst'] , ax = ax2,palette = 'coolwarm')
s.boxplot (x= df['diagnosis'], y = df['concave points_worst'], ax = ax3,palette = 'coolwarm')
s.boxplot (x= df['diagnosis'], y = df['symmetry_worst'], ax = ax4,palette = 'coolwarm')
s.boxplot (x= df['diagnosis'], y = df['fractal_dimension_worst'], ax = ax5,palette = 'coolwarm')
f .tight_layout()

Function for Model fitting and best parameter values 
# Functionalize Model Fitting 
# User Randon seed to get the same value everywhere 
n_jobs : int, default=None
    Number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.


def FitModel (X,Y, algo_name , algorithm, gridSearchParams, cv):
    np.random.seed(10)
    x_train, x_test, y_train, y_test = train_test_split (X,Y,test_size = 0.2)
    
    # Find the Parameters , then choose best parameters 

    grid = GridSearchCV(estimator = algorithm, param_grid = gridSearchParams,
                        cv = cv, scoring = 'accuracy', verbose = 1 , n_jobs = -1 )
    
    grid_result = grid.fit(x_train, y_train)
    best_params = grid_result.best_params_
    pred = grid_result.predict (x_test)
    cm = confusion_matrix (y_test,pred)
    
    print (pred)
    pickle.dump(grid_result,open(algo_name,'wb'))
    
    print ('Best Params :', best_params)
    print ('Classification Report:',classification_report(y_test,pred))
    print ('Accuracy Score', (accuracy_score(y_test,pred)))
    print ('Confusion Matrix :\n',cm)
        
Create SVM Model
#cv = cross validation 
param = {
            'C': [0.1,1,100,1000],
            'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }

FitModel (x,y,'SVC',SVC(), param, cv =10)

Create Random Forest 
param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (x,y,'Random Forest',RandomForestClassifier(), param, cv =10)

Create Random Forest Normal Way 
np.random.seed(10)
x_train,x_test, y_train,y_test = train_test_split (x,y,test_size = 0.2)
forest = RandomForestClassifier (n_estimators = 500)
fit = forest.fit (x_train, y_train)
accuracy = fit.score(x_test,y_test)
predict = fit.predict(x_test)
cmatrix = confusion_matrix (y_test, predict)
print ('Classification Report:',classification_report(y_test,predict))
print ('Accuracy Score', (accuracy_score(y_test,predict)))
print ('Accuracy of Random Forest ', (accuracy))
print ('Confusion Matrix :\n',cmatrix)

XG Boost 
param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (x,y,'XGBoost', XGBClassifier(),param, cv = 10)



Balancing the Data 
Install the Following packages in Anaconda 
# pip install imblearn
Over Sampling Algorithm 
from imblearn.over_sampling import SMOTE

Data Split 
display (df['diagnosis'].value_counts())

Over Sampling 
sm = SMOTE(random_state =42)
X_res, Y_res = sm.fit_resample (x, y)

Print Count 
display (Y_res.value_counts())

Create Random Forest Model 
param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res, Y_res ,'Random Forest',RandomForestClassifier(), param, cv =10)

Create SVC Model  

param = {
            'C': [0.1,1,100,1000],
            'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }
FitModel (X_res, Y_res,'SVC',SVC(), param, cv =10)


XG Boost Model 
param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res, Y_res,'XGBoost', XGBClassifier(),param, cv = 10)

Feature Selection 
Feature Importance /Feature Selection 
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
print ("Feature Ranking:")
for f in range (x.shape[1]):
    print ("Feature %s (%f)"  %(list (x)[f],importances[indices[f]]))

Feature Selection Chart 
feat_imp = pd.DataFrame({'Feature': list(x), 'Gini importance': importances[indices]})
plt.rcParams['figure.figsize']= (12,12)
s.set_style ('whitegrid')
ax= s.barplot(x ='Gini importance', y = 'Feature', data = feat_imp  )
ax.set (xlabel = 'Gini Importances')
plt.show()

feat_imp.index = feat_imp.Feature

Get first 15 Columns 
feat_to_keep = feat_imp.iloc[:15].index
display (type(feat_to_keep),feat_to_keep)
Create Random Forest Model 
X_res = pd.DataFrame(X_res)
Y_res = pd.DataFrame(Y_res)
X_res.columns = x.columns
param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res [feat_to_keep], Y_res ,'Random Forest',RandomForestClassifier(), param, cv =10)

Create SVM Forest Model 

param = {
            'C': [0.1,1,100,1000],
            'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }
FitModel (X_res [feat_to_keep], Y_res,'SVC',SVC(), param, cv =5)

Create XGBOOST  Model 

param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res [feat_to_keep], Y_res,'XGBoost', XGBClassifier(),param, cv = 5)

Create the model with resample data 

Random Forest Classifier  
param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res, Y_res ,'Random Forest',RandomForestClassifier(), param, cv =10) 
Support Vector Machine 
  param = {
            'C': [0.1,1,100,1000],
            'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }
FitModel (X_res, Y_res,'SVC',SVC(), param, cv =10)

XG Boost Classifier 

param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res, Y_res,'XGBoost', XGBClassifier(),param, cv = 10)

Load Pickle file XG Boost  

load_model =pickle.load(open("XGBoost","rb"))

Prediction   
pred1 = load_model.predict (x_test)
print (pred1)

Best Parameter   
load_model.best_params_

Accuracy Score  
print (accuracy_score (pred1,y_test))

Load Pickle file Support Vector Machine   
load_model =pickle.load(open("SVC","rb"))
pred1 = load_model.predict (x_test)
print (load_model.best_params_)
print (accuracy_score (pred1,y_test))
display (pred1)

Load Pickle file Random Forest    

load_model =pickle.load(open("Random Forest","rb"))
pred1 = load_model.predict (x_test)
print (load_model.best_params_)
print (accuracy_score (pred1,y_test))
display (pred1)


Prediction with Another Notebook

Import Libraries 

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os 
import pickle

Load Data Set 

os.chdir ('C:\\Noble\\Training\\Top Mentor\\Training\\Presentation\\Project\\Project to Explain Classifciation\\')
data = pd.read_csv('data.csv')
display (data)

Create X 

x= data.drop ('diagnosis',axis =1).drop('id',axis =1)
display (x)

Create Y

y = data['diagnosis']
display (y)

Update Y to Category Column 

y = y.astype('category')
display (y)

Apply Cat Codes (Label Encoding)

y = y.cat.codes
display (y)

Load XG Boost  and Predict 

load_model =pickle.load(open("XGBoost","rb"))
pred1 = load_model.predict (x)
print (load_model.best_params_)
print (accuracy_score (pred1,y))
display (pred1)

Load Support Vector Machine 

load_model =pickle.load(open("SVC","rb"))
pred1 = load_model.predict (x)
print (load_model.best_params_)
print (accuracy_score (pred1,y))
display (pred1)

Load Random Forest 

load_model =pickle.load(open("Random Forest","rb"))
pred1 = load_model.predict (x)
print (load_model.best_params_)
print (accuracy_score (pred1,y))
display (pred1)
