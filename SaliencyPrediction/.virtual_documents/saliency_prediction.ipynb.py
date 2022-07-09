get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")


from data import Cat2000Loader
from model import MLNet
from loss import ModMSELoss
from trainer import Trainer


# Create loaders
import torch
from torchvision.transforms import Compose, Resize, ToTensor

ratio = 1080/1920
width = 1280
image_size = (int(width*ratio) , width)
# image_size = (480, 640)
fix_map_size = (image_size[0] // 8, image_size[1] // 8)
prior_size = (fix_map_size[0] // 10, fix_map_size[1] // 10)

transform1 = Compose([Resize(image_size), ToTensor()])
transform2 = Compose([Resize(fix_map_size), ToTensor()])

loaders = Cat2000Loader('cat2000', batch_size=5, transform=transform1, target_transform=transform2)


# Prepare model
model = MLNet(prior_size)

criterion = ModMSELoss(*fix_map_size)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,weight_decay=0.0005,momentum=0.9,nesterov=True)


# Train
trainer = Trainer(model, criterion, optimizer, loaders)

trainer.run_trainer(1000)


# Save model
import datetime

full_path = 'models/' + str(datetime.datetime.now()) + '_' + '.basic_model'
torch.save(model.state_dict(), full_path)


from utils import display_evolution


display_evolution()



