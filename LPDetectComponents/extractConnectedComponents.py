from PIL import Image
import glob
from tqdm import trange
from scipy import ndimage
import numpy as np

THRESHOLD = 230
MIN_AREA = 20

images = []
counter = 0
images_files = list(glob.iglob("../images/greyClosingImages/*"))
init_image_files = list(glob.iglob("../images/startImages/*"))
new_counter = 0
for counter in trange(len(images_files)):
    image = np.array(Image.open(images_files[counter]))
    labeled, _ = ndimage.label(image > THRESHOLD)
    cc = ndimage.find_objects(labeled)
    for ccc in cc:
        if len(image[ccc]) > MIN_AREA:
            Image.fromarray(image[ccc]).save(
                f"images/extractedComponentsImages/{new_counter}.jpeg")
            new_counter += 1
