from flask import Flask, request, render_template
from io import BytesIO
import cv2
import numpy as np
from detect import detect_object

app = Flask(__name__)

@app.route('/analyse', methods=['POST'])
def analyse():
    file_bytes = BytesIO(request.files['image'].read())
    img = np.frombuffer(file_bytes.getvalue(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return detect_object(img)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
