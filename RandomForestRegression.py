#Importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

#Loading the dataset and splitting it to feature and target variables (x, y)
dataset = pd.read_csv('/home/naturalcapsule/Downloads/archive(1)/Clean_Dataset.csv')
dataset = dataset.drop(['Unnamed: 0', 'flight'], axis = 1)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Label encode the 'class' column
le = LabelEncoder()
x[:, 6] = le.fit_transform(x[:, 6])


#OneHotEncode the 'source_city, departure_time, stops, arrival_time, destination_city'
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1, 2, 3, 4, 5])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x).toarray())

#Splitting the feature and the target data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 0)

#Building a RandomForestRegression model to predict the model and make predictions with it
regressor = RandomForestRegressor(random_state = 0, n_estimators = 10, n_jobs = 10, criterion = 'squared_error')
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

#Printing out the accuracy
print(r2_score(y_test, y_pred))