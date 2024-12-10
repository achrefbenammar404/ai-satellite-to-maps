from flask import Flask, request, send_file
from keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the generator model
MODEL_PATH = "../model/model_latest.h5"  # Adjust path if needed
model = load_model(MODEL_PATH)

def preprocess_image(img):
    img = img.convert('RGB').resize((256,256))
    img = np.asarray(img)
    img = (img - 127.5) / 127.5
    return np.expand_dims(img, 0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    img = Image.open(file)
    input_img = preprocess_image(img)
    gen_image = model.predict(input_img)
    # scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0
    gen_image = np.clip(gen_image[0], 0, 1)
    gen_image = Image.fromarray((gen_image*255).astype('uint8'))
    
    buf = io.BytesIO()
    gen_image.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
