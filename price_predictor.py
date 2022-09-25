from random import random
from webbrowser import get
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
#over fitting and underfitting 

def get_mae(max_leaf_nodes,train_x,val_x,train_y,val_y):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(train_x,train_y)
    preds_vals=model.predict(val_x)
    mae=mean_absolute_error(val_y,preds_vals)
    return mae

'''for max_leaf_nodes in [5,50,500,5000]:
    my_mae=get_mae(max_leaf_nodes,train_x,val_x,train_y,val_y)

    print("Max leaf nodes :%d \t\t Mean absolute Error :%d"%(max_leaf_nodes,my_mae))'''
candidate_max_leaf_nodes=[5,25,50,100,250,500]

solution={leaf_size:get_mae(leaf_size,train_x,val_x,train_y,val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size=min(solution,key=solution.get)
print(best_tree_size)
print(solution)

'''The random forest model
The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. 
It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. 
If you keep modeling, you can learn more models with even better performance, 
but many of those are sensitive to getting the right parameters.
'''
from sklearn.ensemble import RandomForestRegressor
forest_model=RandomForestRegressor(random_state=1)
forest_model.fit(train_x,train_y)
forest_predicts=forest_model.predict(val_x)
print("Here is the mean_absolute_erro for random_forest")
'''See here the mean absolute error reduces then using the random 
forest model '''
print(mean_absolute_error(val_y,forest_predicts))

