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
from keys import ACCESS_ID, ACCESS_KEY

from flask import Flask, request, jsonify, send_file

app = Flask(__name__)


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def download(file_name, bucket):
    if not os.path.exists("data"):
        os.makedirs("data")
    try:
        s3 = boto3.resource('s3', aws_access_key_id=ACCESS_ID, aws_secret_access_key=ACCESS_KEY)
        output = f"data/{os.path.basename(file_name)}"
        s3.Bucket(bucket).download_file(remove_prefix(file_name, f"s3://{bucket}/"), output)
        return output
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

@app.route('/product_ai', methods=["POST", "GET"])
def get_ai_prediction():
    
    if request.method == "POST":

        bucket_name = request.json["bucket"]

        # 1. Download files 
        ## Receive one s3 image url
        if request.form.get("s3_path"):
            s3_path = request.form.get("s3_path")
            file_path = download(s3_path, bucket_name)

        ## Receive an array of s3 image url
        # sample:
        #  {
        #     "s3_path":["s3://dimuto-live/jobs/image.jpg","s3://dimuto-live/jobs/image2.jpg"]
        # }
        elif request.json:
            json = request.json

            # for each image, download from s3
            for s3_path in json["s3_path"]:
                file_path = download(s3_path, bucket_name)
            file_path = os.path.dirname(file_path)

        ## Receive image file directly
        elif request.files["image"]:
            file = request.files["image"] # works for jpg -> check if can run with jpg
            filename = file.filename
            file_path = os.path.join(data_dir, filename)
            file.save(file_path)

        # 2. Run Product Quality AI
        defect_acceptance_level, pq_score = ai_pipeline(file_path)
        
        # 3. Remove downloaded files
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

        # 4. Return a json output
        # sample: 
        # {
        #     "defect_acceptance_level": "acceptable",
        #     "pq_score": 5
        # }
        return jsonify({
            "defect_acceptance_level" : defect_acceptance_level,
            "pq_score" : pq_score
        })

        

if __name__ == '__main__' :
    # Development
    app.run(debug=True)

    # # Production
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=80)