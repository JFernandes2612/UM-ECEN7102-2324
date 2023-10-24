from PIL import Image, ImageFilter, ImageOps
import glob
from tqdm import trange
from scipy import signal
import numpy as np

def conv_char(img_path):
    image = Image.open(img_path)
    grey_scale_image = ImageOps.autocontrast(image.convert('L').filter(ImageFilter.GaussianBlur(1)))
    new_img = np.array(grey_scale_image) > 210

    new_img = np.array(ImageOps.invert(Image.fromarray(new_img)))

    threshold = 15
    x_lines = signal.find_peaks(-new_img.sum(axis=0))[0]
    x_lines = x_lines[new_img.sum(axis=0)[x_lines]<threshold]

    images = []

    for i, _ in enumerate(x_lines):
        if i==0:
            images.append(Image.fromarray(np.array(image)[:,:x_lines[i]]))
        else:
            images.append(Image.fromarray(np.array(image)[:,x_lines[i-1]:x_lines[i]]))

    return images

def main():
    images_files = list(glob.iglob("../images/labeled/LP/*"))
    new_counter = 0
    for counter in trange(len(images_files)):
        conv_images = conv_char(images_files[counter])
        for image in conv_images:
            image.save(f"../images/characters/{new_counter}.jpeg")
            new_counter += 1
       

if __name__ == "__main__":
    main()