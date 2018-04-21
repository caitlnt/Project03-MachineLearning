from __future__ import print_function
from flask import Flask, render_template, jsonify, redirect, request
import sagemaker
from sagemaker.mxnet import MXNetPredictor
from sagemaker.tensorflow import TensorFlowPredictor
import sys
import ast

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process')
def process():
    return render_template('process.html')

@app.route('/application')
def application():
    return render_template('application.html')

@app.route('/visuals')
def visuals():
    return render_template('visuals.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/conclusions')
def conclusions():
    return render_template('conclusions.html')

@app.route('/appendix')
def appendix():
    return render_template('appendix.html')
    
   
if __name__ == "__main__":
    app.run(debug=True)
