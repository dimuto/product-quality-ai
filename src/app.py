import os
import shutil
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
data_dir = os.path.join(ROOT, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

from run import ai_pipeline

from flask import Flask, request

app = Flask(__name__)


@app.route('/product_ai', methods=['POST', 'GET'])
def get_ai_prediction():
    if request.method == 'POST':

        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(data_dir, filename)
        file.save(file_path)

        defect_acceptance_level, pq_score = ai_pipeline(file_path)
        # shutil.rmtree(file_path)
        return str(pq_score)
        

if __name__ == '__main__' :
    app.run(debug=True)