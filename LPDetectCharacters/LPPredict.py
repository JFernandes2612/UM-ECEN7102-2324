from LPDetectCharacters.LPExtractFeatures import conv_char
import numpy as np
from skimage import transform
import tensorflow as tf
import os
import glob
from PIL import Image

def load_image(img):
    np_image = np.array(img).astype('float32')
    np_image = transform.resize(np_image, (64, 48, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def predict(images):
    model = tf.keras.models.load_model('models/character.tf')

    print(model.summary())

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                   'J', 'K', 'L', 'M', 'N', 'None', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    prediction_val = []

    for i, img in enumerate(images):
        np_image = load_image(img)
        prediction = model.predict(np_image)
        prediction_list = prediction.tolist()[0]
        max_value = max(prediction_list)
        max_index = prediction_list.index(max_value)
        print(
            f"Prediction result for image {i}: {class_names[max_index]} with {max_value*100}% certainty")
        
        prediction_val.append(class_names[max_index] if class_names[max_index] != "None" else " ")

    return "".join(prediction_val)

def main():
    files = glob.glob('test_results/*')
    for f in files:
        os.remove(f)
    images = conv_char("test/1.jpg")
    for i, img in enumerate(images):
        img.save(f"test_results/{i}.jpg")
    predict(images)


if __name__ == "__main__":
    main()
