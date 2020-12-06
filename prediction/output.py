import pandas as pd 
from clean import clean_data


# This is the last function that where I introduce the model I selected to export the csv with my prediction

def outputs (model,pred,df,csv_path_name):
    df = clean_data(df)
    pred = clean_data(pred)
    y = df['price']
    x = df.drop(columns='price')
    model = model()
    model.fit(x,y)
    y_pred = (model.predict(pred))
    return pd.DataFrame(y_pred, columns=["price"]).to_csv(csv_path_name)