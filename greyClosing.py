from PIL import Image
import glob
from tqdm import trange
from scipy import ndimage


images = []
counter = 0
images_files = list(glob.iglob("images/edgeDetectionImages/*"))
for counter in trange(len(images_files)):
    image = Image.open(images_files[counter])
    closing = ndimage.grey_closing(image, size=(3, 17))
    Image.fromarray(closing).save(f"images/greyClosingImages/{counter}.jpeg")
    
