import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def main():

    if request.method == "POST":

        clf = joblib.load("clf.pkl")
        h = request.form.get("height")
        w = request.form.get("weight")

        X = pd.DataFrame([[h, w]],
                         columns=["Height","weight"])
        pred = clf.predict(X)

    else:
        pred = ""

    return render_template("website.html",output = pred)


if __name__ == '__main__':
    app.run(debug=True)
