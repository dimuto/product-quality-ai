import flask
from PIL import Image
import io
from flask import Flask, request

import numpy as np
import pandas as pd

from detect import run 

app = Flask(__name__)

# text - http://127.0.0.1:5000/?length=60
@app.route("/", methods=["GET"])
def predict():
    length = request.args.get("length")
    prediction = int(length) * int(length)
    return str(prediction)

# file - use postman
@app.route("/predict_file", methods=["POST"])
def predict_file():
    input_data = pd.read_csv(request.files.get("input_file"))
    prediction = list(input_data) * 10
    return str(list(prediction))

# image
@app.route("/product_ai", methods=["POST"])
def predict_defect():
    run(source=request.files.get("input_file"))

if __name__ == '__main__':
    app.run()
