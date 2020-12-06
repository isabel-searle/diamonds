import pandas as pd



def clean_data(df):
    cut_values={'Fair':5, 'Good':4, 'Very Good':3, 'Premium':2, 'Ideal':1}
    clarity_values={'I1':8,'SI2':7,'SI1':6, 'VS2':5, 'VS1':4, 'VVS2':3, 'VVS1':2, 'IF':1}
    color_values={'D':1, 'E':2, 'F':3, 'G':4, 'H':5, 'I':6, 'J':7}
    df['cut']=df['cut'].map(cut_values)
    df['clarity']=df['clarity'].map(clarity_values)
    df['color']=df['color'].map(color_values) 
    return df