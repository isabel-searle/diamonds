# diamonds

# 1. What am I trying to predict?

The goal of this proyect is to predict the prices of diamonds given some features. 

According to the data and the goal I need to use a regression model because I'm trying to predict a continuous dependent variable (a price) from a number of independent variables.

# 2. Let's have a look at the data

First of all I check the types. I have 3 object type features. One of the solution could have been using get_dummies() but the data_set has a description with a categorical **ordinal** stipulation so I use LabelEncoder or map() to give the corresponding value.

Then I check the colinearity between the features:

<p><img src="images\heatmap.png" width="90%">

'x', 'y', 'z', and 'carat' are correlated because this is the volume and weight of the diamond. They are, of course, correlated with the price but any way I will check what happens if I remove 'x', 'y', 'z' that doesn't provide more information than the weight (all 4 features have almost the same correlation with price and they are directly related between themselfs, so the colinearity is hugh).

Also I check with SelectKBest and this feature selection from sklearn suggests to remove 'depth'.

I'm not still sure, so I will test with and without the questionable features.

# 3. Testing some models

I created this the function xyz_test to check if I should keep or not xyz, I checked all the models one by one and finally I improved the function with all models and change of features in ones, but SGDRegressor and SVR are too long process for a very bad result, so I leave them printed in the notebook, but I didn't add them to the final function:

<p><img src="images\xyz_test.png" width="90%">

I used test_features function to decide the top 3 model with the corresponding top 2 features.

The ranking model is as follows:

1. RandomForestRegressor: no_depht, all_f

2. DecisionTreeRegressor: no_depth_no_xyz, no_xyz

3. KNeighborsRegressor: no_depth, all_f

<p><img src="images\top_features.png" width="90%">


# 4. Prediction

I finally make my prediction with the output function defined whose goal is use the model I chose to predict with the predict.csv dataset and creat the new csv with the predicted prices.

# 5. Files

All files are in the same directory becuase otherwise I wouldn't be able to import some functions others.

Draft with the justification of my decisions:

- draft_before_functions.ipynb

Defined functions:

- clean.py
- output.py
- tests.py

Pipeline:

- pipeline.ipynb

