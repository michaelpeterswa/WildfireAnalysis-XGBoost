# First XGBoost model for Washington Wildfires dataset
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import csv
from pandas import DataFrame
from collections import defaultdict
import numpy 

header = ["date","county","cause","binlat","binlon"]
# load data
X = []
Y = []
with open('data/clean_fire_data.csv') as csvfile:
    reader = csv.reader(csvfile)
    for fire in reader:
        X.append([fire[0], fire[2], fire[3], fire[6], fire[7]])
        Y.append(int(fire[8]))

df = DataFrame(X, columns=header)
encoder_dict = defaultdict(LabelEncoder)
labeled_df = df.apply(lambda x: encoder_dict[x.name].fit_transform(x))
# print(labeled_df)

inverse_transform_lambda = lambda x: encoder_dict[x.name].inverse_transform(x)
# print(labeled_df.apply(inverse_transform_lambda))

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(labeled_df, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [numpy.round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))