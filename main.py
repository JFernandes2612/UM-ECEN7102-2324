from PIL import Image, ImageFilter, ImageOps
import glob
from tqdm import trange
from scipy import ndimage
import numpy as np

THRESHOLD = 230
MIN_AREA = 20

images = []
counter = 0
images_files = list(glob.iglob("images/startImages/*"))
new_counter = 0
for counter in trange(len(images_files)):
    image = Image.open(images_files[counter])
    grey_scale_image = image.convert('L')
    edges_image = ImageOps.autocontrast(grey_scale_image.filter(ImageFilter.FIND_EDGES))
    grey_closing_images = ndimage.grey_closing(edges_image, size=(3, 17))
    labeled, _ = ndimage.label(grey_closing_images > THRESHOLD)
    cc = ndimage.find_objects(labeled)
    for ccc in cc:
        if len(grey_closing_images[ccc]) > MIN_AREA:
            Image.fromarray(np.array(image)[ccc]).convert('RGB').save(f"images/extractedComponentsImages/{new_counter}.jpeg")
            new_counter += 1
    
