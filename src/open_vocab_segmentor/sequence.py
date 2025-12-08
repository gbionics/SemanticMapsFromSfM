import os
from PIL import Image

class Sequence(object):
    def __init__(self, path, images_folder="images") -> None:
        self.path = os.path.join(path, images_folder)
    
    def load_images(self):
        image_dirlist = sorted(os.listdir(self.path))
        images = [Image.open(os.path.join(self.path, image_name)) for image_name in image_dirlist]
        return images, image_dirlist