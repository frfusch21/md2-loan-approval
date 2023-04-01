from flask import Flask, render_template, request, redirect, make_response
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/predict", methods=['GET', 'POST'])
def pred():
    df = pd.DataFrame(columns=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area'])

    if request.method == "POST":
        name = request.form["name"]
        gender = request.form["gender"]
        marriage = request.form["marriage"]
        dependent = request.form["dependent"]
        education = request.form["education"]
        self_employed = request.form["self_employed"]
        ap_income = request.form["ap_income"]
        coap_income = request.form["coap_income"]
        loan_amount = request.form["loan_amount"]
        loan_amount_term = request.form["loan_amount_term"]
        credit_history = request.form["credit_history"]
        property_area = request.form["property_area"]

        df = df.append({'Gender': gender, 'Married': marriage, 'Dependents': dependent, 'Education': education,'Self_Employed': self_employed, 'ApplicantIncome': ap_income, 'CoapplicantIncome':coap_income, 'LoanAmount':loan_amount, 'Loan_Amount_Term':loan_amount_term, 'Credit_History':credit_history, 'Property_Area':property_area}, ignore_index=True)
        with open('model/Random_Forest_model.pkl','rb') as file:
            Random_Forest_model = pickle.load(file)

        prediction = Random_Forest_model.predict(df)

        with open('data_collection.txt','a') as file:
            file.write("%s\n" % df)
    
        return render_template('result.html', pred=str(prediction[0]), name=name)

if __name__ == "__main__":
    app.debug=True
    app.run()