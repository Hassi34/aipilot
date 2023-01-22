import os
from pathlib import Path
from PIL import Image

def bulk_resize(dataset_dir, width , height):
    dataset_dir = Path(dataset_dir)
    for dirpath, dirs, files in os.walk(dataset_dir):
        for filename in files:
            fname = os.path.join(dirpath,filename)
            if fname.endswith(('.jpg','.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
                img = Image.open(fname)
                resized_img = img.resize((width, height))
                resized_img.save(fname)