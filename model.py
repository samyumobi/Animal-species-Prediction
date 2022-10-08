import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib

df = pd.read_csv("data.csv")

# Peek at the dataset contents
print(df.head())

# Split the dataset into x - train set, y-test set
x = df[["Height", "Weight"]]
y = df["Species"]

# Utilise Naive bayes algorithm
clf = GaussianNB()
clf.fit(x, y)

# Store the present state of trained model
joblib.dump(clf, "clf.pkl")
