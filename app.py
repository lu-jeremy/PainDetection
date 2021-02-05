from flask import *
from PainNet2D import PainDetector2DNet
import nibabel as nib
import os
import numpy as np
from werkzeug.utils import secure_filename
from keras import backend
import requests

# del test.py

pain_det = PainDetector2DNet()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if len(request.files) != 0:
            data = request.files['file']
            data_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(data.filename))
            data.save(data_path)

            img = nib.load(data_path)
            img_data = img.get_fdata()

            pred = pain_det.predict(img_data)

            print(np.mean(pred))

            if np.mean(pred) >= .5:
                pred = 1
                requests.get('http://192.168.1.42:5002/action/start')
            else:
                pred = 0
                requests.get('http://192.168.1.42:5002/action/stop')

    backend.clear_session()

    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
