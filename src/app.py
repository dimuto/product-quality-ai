import os
import shutil
from pathlib import Path

import boto3
import botocore

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
data_dir = os.path.join(ROOT, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

from run import ai_pipeline

from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

BUCKET_NAME = "product-quality-ai" 
# # Test
# FILENAME = "defect_raw/defect_goods_received/2021-07-26-11-05-37-1627272405.jpg"


def download(file_name, bucket):
    try:
        s3 = boto3.resource('s3')
        output = f"data/{os.path.basename(file_name)}"
        s3.Bucket(bucket).download_file(file_name, output)
        return output
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


@app.route('/product_ai', methods=["POST", "GET"])
def get_ai_prediction():
    
    if request.method == "POST":

        ## Receive s3 image url
        if request.form.get("s3_path"):
            s3_path = request.form.get("s3_path")
            file_path = download(s3_path, BUCKET_NAME)

        ## Receive image file directly
        elif request.files["image"]:
            file = request.files["image"] # check if can run with jpg
            filename = file.filename
            file_path = os.path.join(data_dir, filename)
            file.save(file_path)

        defect_acceptance_level, pq_score = ai_pipeline(file_path)
        
        if os.path.isfile(file_path):
            os.remove(file_path)

        return jsonify({
            "defect_acceptance_level" : defect_acceptance_level,
            "pq_score" : pq_score
        })


    # ## Send images 
    # if request.method == "GET":
    #     output = download(FILENAME, BUCKET_NAME)
    #     return send_file(output, as_attachment=True)

    ## integration with DPL
    # authorisation for username and pw
        

if __name__ == '__main__' :
    app.run(debug=True)