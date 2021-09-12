import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns 

#Q1

print("NumPy version:{}".format(np.__version__))


#Q2
 
print("Pandas version:{}".format(pd.__version__))


#Q3

cars_df = pd.read_csv(r"C:\Users\Daria\car_dataset.csv")
df=cars_df.copy(deep=True)

df_bmw=cars_df.loc[cars_df['Make']=="BMW"]
avg_bmw=round(df_bmw['MSRP'].mean())
print("Average price of BMW cars:{}".format(avg_bmw))

#Q4

cars_2015=df.loc[df['Year']>=2015]
nan_engine=cars_2015["Engine HP"].isna().sum()
print("NaN values of Engine HP:{}".format(nan_engine))

#Q5

avg_engine=round(cars_df['Engine HP'].mean())
print("Average Engine HP:{}".format(avg_engine))

df['Engine HP'].fillna(avg_engine)
print("New Enigine HP mean:{}".format(round(df['Engine HP'].mean())))

#Q6

df_royce=df.loc[df['Make']=="Rolls-Royce"]
df_royce_small=df_royce[["Engine HP", "Engine Cylinders", "highway MPG"]].drop_duplicates()
print(df_royce_small)

x=df_royce_small.to_numpy()
xt=x.transpose()
xtx=xt.dot(x)

inv_xtx=np.linalg.inv(xtx)
print(inv_xtx)

sum_inv= np.sum(inv_xtx)
print("Sum of matrix elements:{}".format(sum_inv))

#Q7

y=np.array([1000, 1100, 900, 1200, 1000, 850, 1300])
inv_xtx_xt= inv_xtx.dot(xt)
w=inv_xtx_xt.dot(y)
print("First element of w is:{}".format(w[0]))
