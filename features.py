from PIL import Image, ImageFilter, ImageOps
import glob
from tqdm import trange
from scipy import ndimage
import numpy as np


THRESHOLD = 230
MIN_AREA = 20


def conv_image(img_path):
    image = Image.open(img_path)
    grey_scale_image = image.convert('L')
    edges_image = ImageOps.autocontrast(
        grey_scale_image.filter(ImageFilter.FIND_EDGES))
    grey_closing_images = ndimage.grey_closing(edges_image, size=(3, 17))
    labeled, _ = ndimage.label(grey_closing_images > THRESHOLD)
    cc = ndimage.find_objects(labeled)
    images = []
    for ccc in cc:
        if len(grey_closing_images[ccc]) > MIN_AREA:
            images.append(Image.fromarray(np.array(image)[ccc]).convert('RGB'))
    return images


def main():
    images = []
    counter = 0
    images_files = list(glob.iglob("images/startImages/*"))
    new_counter = 0
    for counter in trange(len(images_files)):
        conv_images = conv_image(images_files[counter])
        for image in conv_images:
            image.save(f"images/extractedComponentsImages/{new_counter}.jpeg")
            new_counter += 1

if __name__ == "__main__":
    main()