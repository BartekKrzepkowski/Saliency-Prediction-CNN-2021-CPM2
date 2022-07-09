import os
import glob
import ipyplot
import numpy as np
from PIL import Image

def draw_at_random(path):
    drawed_imgs = {}
    for category in os.listdir(imgs_path):
        candidates = [img for img in os.listdir(os.path.join(imgs_path, category)) if img.endswith('.jpg')]
        drawed_imgs[category] = os.path.join(imgs_path, category, np.random.choice(candidates))
    return drawed_imgs

def display_evolution():
    root = lambda x: f'cat2000/selected/{x}/'
    for category in os.listdir(root('images')):
        original_image = Image.open(glob.glob(os.path.join(root('images'), category, '*.jpg'))[0])
        fix_map = Image.open(glob.glob(os.path.join(root('maps'), category, '*.jpg'))[0])
        print(f'CATEGORY: {category}')
        ipyplot.plot_images([original_image, fix_map], ['original image', 'map fixation'], img_width=500)
        pred = [Image.open(path) for path in glob.glob(os.path.join(root('maps'), category, 'outputs', '*.jpeg'))]
        ipyplot.plot_images([img for i, img in enumerate(pred) if i%10 == 9], 10*np.arange(10)+9, img_width=300)