# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 23:19:37 2020

@author: Eslam Youssef
"""

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('classifier.pkl', 'rb'))
scaler= pickle.load(open('scaler.pkl','rb'))

#print('Model Attributes')
#print(model.__dict__)
#print('Scaler Attributes')
#print(scaler.__dict__)
#print('\n\n Age Mean=',scaler.mean_[0])
#print('\n\n Age Std=',scaler.scale_[0])
#print('\n\n Salary Mean=',scaler.mean_[1])
#print('\n\n Salary Std=',scaler.scale_[1])

age_mean=scaler.mean_[0]
age_std=scaler.scale_[0]
salary_mean=scaler.mean_[1]
salary_std=scaler.scale_[1]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    print('Age Mean=',age_mean)
    
    
    
    data = request.form.to_dict()
    
    age=int(data['age'])
    salary=int(data['salary'])
    
    # Scaling inputs
    age_scaled=float((age-age_mean)/age_std)
    salary_scaled=float((salary-salary_mean)/salary_std)
    
    print('Age Scaled: ',age_scaled)
    print('Salary Scaled: ',salary_scaled)

    
    arr=np.array([[age_scaled,salary_scaled]])

    
    pred=model.predict(arr)
    
    return render_template('index.html', result=pred)

if __name__ == "__main__":
    app.run(debug=True)