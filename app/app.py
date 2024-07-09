from flask import Flask, request, render_template
import torch
from torchvision import transforms as T
from PIL import Image
import io
import json
from src.inference import get_prediction

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        boxes, labels = get_prediction(img, threshold=0.8)
        result = {"boxes": boxes, "labels": labels}
        return render_template('result.html', result=result)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
