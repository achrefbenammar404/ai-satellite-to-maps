import argparse
from keras.models import load_model
import numpy as np
from PIL import Image

def load_image(filename, size=(256,256)):
    image = Image.open(filename).convert('RGB')
    image = image.resize(size)
    pixels = np.asarray(image)
    pixels = (pixels - 127.5) / 127.5
    return np.expand_dims(pixels, 0)

def save_image(image_array, filename):
    # scale back from [-1,1] to [0,1]
    image = (image_array + 1) / 2.0
    image = np.clip(image[0], 0, 1)
    image = Image.fromarray((image*255).astype('uint8'))
    image.save(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help="Path to the trained generator model")
    parser.add_argument('--input_image', required=True, help="Path to input satellite image")
    parser.add_argument('--output_image', default='output.png', help="Output file name")
    args = parser.parse_args()

    model = load_model(args.model_path)
    input_img = load_image(args.input_image)
    gen_image = model.predict(input_img)
    save_image(gen_image, args.output_image)
    print(f"Saved generated image to {args.output_image}")
