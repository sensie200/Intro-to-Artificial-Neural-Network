from random import random
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
#reading the csv file
house_data=pd.read_csv("melb_data.csv")
#filtering the data to remove the nas
house_filtered=house_data.dropna(axis=0)
#printing out some of the coumnsl
#print(house_filtered.columns)
y=house_filtered.Price
features=['Rooms','Bathroom','Landsize','Lattitude','Longtitude']
x=house_filtered[features]
#print(x.describe)
#print(x.head(5))
#building the model 
Housing_model=DecisionTreeRegressor(random_state=1)
Housing_model.fit(x,y)

#print("Making prediction for a the following 5 houses ")
'''print(x.head(5))
print("The predictions are :")
print(Housing_model.predict(x.head(5)))
predicted_'''

#model validation 
from sklearn.metrics import mean_absolute_error
predicted_house_price=Housing_model.predict(x)
print(mean_absolute_error(y,predicted_house_price))
#model validation using a split data  using the train test split form the sklearn.model_selection
from sklearn.model_selection import train_test_split
#split data into train and validation data ,for both features and target
train_x,val_x,train_y,val_y=train_test_split(x,y,random_state=1)
split_housing_model=DecisionTreeRegressor()
split_housing_model.fit(train_x,train_y)
split_housing_prediction=split_housing_model.predict(val_x)
print(mean_absolute_error(val_y,split_housing_prediction))


