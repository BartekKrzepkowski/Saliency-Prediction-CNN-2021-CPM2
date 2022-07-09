get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")


from data import Cat2000Loader
from model import MLNet
from loss import ModMSELoss
from trainer import Trainer


# Create loaders
import torch
from torchvision.transforms import Compose, Resize, ToTensor

# ratio = 1080/1920
# width = 320
# image_size = (int(width*ratio) , width)
image_size = (480, 640)
fix_map_size = (image_size[0] // 8, image_size[1] // 8)
prior_size = (fix_map_size[0] // 10, fix_map_size[1] // 10)

transform1 = Compose([Resize(image_size), ToTensor()])
transform2 = Compose([Resize(fix_map_size), ToTensor()])

loaders = Cat2000Loader('cat2000', batch_size=16, transform=transform1, target_transform=transform2)


# Prepare model
model = MLNet(prior_size)

criterion = ModMSELoss(*fix_map_size)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,weight_decay=0.0005,momentum=0.9,nesterov=True)


path = 'models/2021-09-16 05:52:44.712130_.basic_model'
model.load_state_dict(torch.load(path))


# Train
trainer = Trainer(model, criterion, optimizer, loaders)

trainer.run_trainer(100)


import datetime

full_path = 'models/' + str(datetime.datetime.now()) + '_' + '.basic_model'
torch.save(model.state_dict(), full_path)


import os
import glob
import ipyplot
from PIL import Image


root = lambda x: f'cat2000/selected/{x}/'
for category in os.listdir(root('images')):
    original_image = Image.open(glob.glob(os.path.join(root('images'), category, '*.jpg'))[0])
    fix_map = Image.open(glob.glob(os.path.join(root('maps'), category, '*.jpg'))[0])
    print(f'CATEGORY: {category}')
    ipyplot.plot_images([original_image, fix_map], ['original image', 'map fixation'], img_width=500)
    pred = [Image.open(path) for path in glob.glob(os.path.join(root('maps'), category, 'outputs', '*.jpeg'))]
    ipyplot.plot_images([img for i, img in enumerate(pred) if iget_ipython().run_line_magic("10", " == 9], 10*np.arange(10)+9, img_width=300)")


from utils import display_evolution


display_evolution()



