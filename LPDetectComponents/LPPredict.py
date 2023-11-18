from LPDetectComponents.LPExtractFeatures import conv_image
import numpy as np
from skimage import transform
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import glob

def load_image(img):
    np_image = np.array(img).astype('float32')
    np_image = transform.resize(np_image, (64, 192, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def predict(images):
    model = tf.keras.models.load_model('../models/licenseplate.tf')

    print(model.summary())

    for i, img in enumerate(images):
        prediction = model.predict(img)
        prediction_res = prediction.tolist()[0][0]
        if prediction_res >= 0.5:
            print(f"Prediction result for image {i}: Not a License Plate with {prediction_res*100}% certainty")
        else:
            print(f"Prediction result for image {i}: License Plate with {100-prediction_res*100}% certainty")


def main():
    files = glob.glob('test_results/*')
    for f in files:
        os.remove(f)
    images = conv_image("test/2.jpg")
    for i,img in enumerate(images):
        img.save(f"test_results/{i}.jpg")
    images = [load_image(image) for image in images]
    predict(images)

if __name__ == "__main__":
    main()
