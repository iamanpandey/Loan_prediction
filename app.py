from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("loan_model.pkl","rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {

    "Gender": request.form["Gender"],
    "Married": request.form["Married"],
    "Dependents": request.form["Dependents"],
    "Education": request.form["Education"],
    "Self_Employed": request.form["Self_Employed"],
    "ApplicantIncome": float(request.form["ApplicantIncome"]),
    "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
    "LoanAmount": float(request.form["LoanAmount"]),
    "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
    "Credit_History": float(request.form["Credit_History"]),
    "Property_Area": request.form["Property_Area"]

    }


    # same encoding as training

    maps = {
    "Gender":{"Male":1,"Female":0},
    "Married":{"No":0,"Yes":1},
    "Education":{"Not Graduate":0,"Graduate":1},
    "Self_Employed":{"No":0,"Yes":1},
    "Property_Area":{"Rural":0,"Semiurban":1,"Urban":2},
    "Dependents":{"0":0,"1":1,"2":2,"3+":4}
    }


    for col in maps:
        data[col] = maps[col][data[col]]


    input_df = pd.DataFrame([data])


    result = model.predict(input_df)[0]


    if result==1:
        prediction="Loan Approved ✅"
    else:
        prediction="Loan Rejected ❌"


    return render_template("index.html",
                           prediction=prediction)
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

