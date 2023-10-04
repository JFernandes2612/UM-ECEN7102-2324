from PIL import Image
import glob
from tqdm import trange


images = []
counter = 0
images_files = list(glob.iglob("images/startImages/*"))
for counter in trange(len(images_files)):
    Image.open(images_files[counter]).convert('L').save(f"images/greyscaleImages/{counter}.jpeg")
    
