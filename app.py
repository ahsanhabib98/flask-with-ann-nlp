from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import os

app = Flask(__name__)

model = pickle.load(open('ann_nlp_model.pkl', 'rb'))


@app.route('/')
def home():
    template = 'home.html'
    return render_template(template)


def prediction(file_path):
    file = np.array(
        [open(file_path, encoding="utf8", errors='ignore').read()]
    )
    file_type = model.predict(file)
    return file_type[0]


@app.route('/predict', methods=['GET', 'POST'])
def predict_file_type():
    if request.method == 'POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'upload_file', secure_filename(file.filename))
        file.save(file_path)
        result = prediction(file_path)
        print(result)
        return 'The file is {} format'.format(str(result))
    return None


if __name__ == '__main__':
    app.run()
