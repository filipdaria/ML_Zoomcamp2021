import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import model_selection
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge

house_df=pd.read_csv(r"C:\Users\Daria\ml_zoomcamp2021\AB_NYC_2019.csv")
df=house_df[['neighbourhood_group','room_type','latitude','longitude','price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]
df=df.dropna()
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = df.reset_index()
#Q1

print(df['neighbourhood_group'].mode())

df['above_average'] = np.where(df['price']<=152,'1','0')
X= df[['neighbourhood_group','room_type','latitude','longitude','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]
Y=df['above_average']
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X_train, y_train, test_size=0.25,random_state=42)

#Q2

corr_matrix=X_train.corr()
#sns.heatmap(corr_matrix, annot=True)
corr=df.corr()
sns.heatmap(corr, annot=True)
plt.show()

#Q3

mis_neigh= round(sk.metrics.mutual_info_score(X_train['neighbourhood_group'], y_train),2)
#print(mis_neigh)
mis_room=round(sk.metrics.mutual_info_score(X_train['room_type'], y_train),2)
#print(mis_room)

#Q4

df_categ=X_train[['neighbourhood_group','room_type']]
encoder=OneHotEncoder(sparse=False)
onehot = encoder.fit_transform(df_categ)

df_test=X_test[['neighbourhood_group','room_type']]
encod=OneHotEncoder(sparse=False)
one = encod.fit_transform(df_test)
#print(onehot)


df_hot = pd.DataFrame(onehot)
df_test_hot=pd.DataFrame(one)
X_train_copy=X_train.copy()
X_test_copy=X_test.copy()

X_train_copy.drop(['neighbourhood_group','room_type'],inplace=True ,axis=1)
X_test_copy.drop(['neighbourhood_group','room_type'],inplace=True ,axis=1)
frames_train=[X_train_copy,df_hot]
frames_test=[X_test_copy,df_test_hot]
result_train=pd.concat(frames_train,axis=1)
result_test=pd.concat(frames_test,axis=1)

#model = sk.linear_model.LogisticRegression(solver='lbfgs', C=1.0, random_state=42).fit(result_train,y_train)
#print(model)
#prediction=model.predict(result_test)
#print(sk.metrics.accuracy_score(y_test, prediction))

#Q6

np_log = np.log1p(df['price'])

X_trainr, X_testr, y_trainr, y_testr = sk.model_selection.train_test_split(result_train, np_log, test_size=0.2, random_state=42)
X_trainr, X_valr, y_trainr, y_valr = sk.model_selection.train_test_split(X_trainr, y_trainr, test_size=0.25,random_state=42)
print(y_trainr.shape)
print(y_valr.shape)