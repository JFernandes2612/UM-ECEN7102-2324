from PIL import Image, ImageFilter, ImageOps
import glob
from tqdm import trange


images = []
counter = 0
images_files = list(glob.iglob("images/greyscaleImages/*"))
for counter in trange(len(images_files)):
    ImageOps.autocontrast(Image.open(images_files[counter]).filter(ImageFilter.FIND_EDGES)).save(f"images/edgeDetectionImages/{counter}.jpeg")