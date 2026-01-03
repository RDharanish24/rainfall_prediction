

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("/content/Rainfall.csv")

data.head()

data['rainfall'].value_counts()

data.columns

data=data.drop(columns=['day'])

data.columns=data.columns.str.strip()

data.columns

data.head()

data['winddirection']=data['winddirection'].fillna(data['winddirection'].mode()[0])

data['windspeed']=data['windspeed'].fillna(data['windspeed'].median())

data.isnull().sum()

data['rainfall']=data['rainfall'].replace({"yes":1,"no":0})

data.tail()

data.head()

sns.set(style="whitegrid")

plt.figure(figsize=(15,10))
for i,column in enumerate(['pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','windspeed'],1):
  plt.subplot(3,3,i)
  sns.histplot(data[column],kde=True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True,cmap="coolwarm")
plt.title("cor matrix")
plt.show()

data=data.drop(columns=['maxtemp','temparature','mintemp'])

df_majority=data[data["rainfall"]==1]
df_minority=data[data["rainfall"]==0]

df_majority_downsampled=resample(df_majority,replace=False,n_samples=len(df_minority),random_state=42)

df_downsampled=pd.concat([df_majority_downsampled,df_minority])

df_downsampled.head()



df_downsampled=df_downsampled.sample(frac=1,random_state=42).reset_index(drop=True)

x=df_downsampled.drop(columns=["rainfall"])
y=df_downsampled["rainfall"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

rf_model=RandomForestClassifier(random_state=42)
param_grid_rf={
    "n_estimators":[50,100,200],
    "max_features":["sqrt","log2"],
    "max_depth":[None,10,20,30],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4]
}

grid_search_rf=GridSearchCV(estimator=rf_model,param_grid=param_grid_rf,cv=5,n_jobs=-1,verbose=2)
grid_search_rf.fit(x_train,y_train)

best_rf_model=grid_search_rf.best_estimator_
print("best parameters are",grid_search_rf.best_params_)

cv_scores=cross_val_score(best_rf_model,x_train,y_train,cv=5)
print(cv_scores)
print(np.mean(cv_scores))

y_pred=best_rf_model.predict(x_test)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

x_train.head()

input_data={1015,19,95,81,0.0,40,13}
input_df=pd.DataFrame([input_data],columns=['pressure','dewpoint','humidity','cloud','sunshine','winddirection','windspeed'])

prediction=best_rf_model.predict(input_df)
print(prediction)

if prediction[0]==1:
    print("it will rain")
else:
    print("it will not rain")

