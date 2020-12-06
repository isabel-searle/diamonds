import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from numpy import asarray
import math
import statistics
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor




# This was the firs function I created in orde to test some models and get the RMSE:

def model_RMSE(df, model):
    df = clean_data(df)
    y = df['price']
    x = df.drop(columns='price')
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    model = model()
    model.fit(X_train,y_train)
    y_pred = (model.predict(X_test))
    return math.sqrt(mean_squared_error(y_test, y_pred))



# I create this function as a pipeline where the dataset is cleaned, devided, modeled 
# and it returns the corresponding metrics:

def pipeline(df, model):
    df = clean_data(df)
    y = df['price']
    x = df.drop(columns='price')
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    model = model()
    result = cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=10).mean()
    model.fit(X_train,y_train)
    y_pred = (model.predict(X_test))
    result = {"cross_val_score_mean":result,"RMSE":math.sqrt(mean_squared_error(y_test, y_pred)),
            'R2':r2_score(y_test, y_pred)}
    return pd.DataFrame(result, index=[0])


# This is a bigger fonction to compare the six models I chose after testing with the lasts functions.

def test_features(df,iterations):
    models = [LinearRegression,DecisionTreeRegressor,ElasticNet,Ridge,Lasso,KNeighborsRegressor,RandomForestRegressor]
    result = {}
    for model in models:
        all_f = []
        no_xyz = []
        no_depth = []
        no_depth_no_xyz = []
        for i in range(iterations):
            df = pd.read_csv('data/train.csv').drop(columns=['id'])
            df_no_xyz=df.drop(columns=['x','y','z'])
            df_no_depth=df.drop(columns=['depth'])
            df_no_depth_no_xyz=df.drop(columns=['depth','x','y','z'])
            all_f.append(model_RMSE(df,model))
            no_xyz.append(model_RMSE(df_no_xyz,model)) 
            no_depth.append(model_RMSE(df_no_depth,model)) 
            no_depth_no_xyz.append(model_RMSE(df_no_depth_no_xyz,model))
            result[model]={"all_f":statistics.mean(all_f), "no_xyz":statistics.mean(no_xyz),
                           "no_depth":statistics.mean(no_depth),"no_depth_no_xyz":statistics.mean(no_depth_no_xyz)}       
    return pd.DataFrame(result).T


# This was the first version of the previews finction. Before defining the function test_features I made 
# the tests one by one, SGDRegressor and SVR were too long process for a very bad result, so I left them 
# printed in the jupyter notebook, but I didn't add them to the final function:
def xyz_test(df,df_drop_xyz,model):
    xyz=[]
    no_xyz=[]
    for i in range(100):
        xyz.append(model_RMSE(df,model))
        no_xyz.append(model_RMSE(df_drop_xyz,model))   
        return {"xyz":statistics.mean(xyz), "no_xyz":statistics.mean(no_xyz)}

