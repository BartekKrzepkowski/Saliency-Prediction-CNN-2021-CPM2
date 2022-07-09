import os
import torch
from PIL import Image
from torchbearer.cv_utils import DatasetValidationSplitter
from torch.utils.data import Dataset, DataLoader

class Cat2000Loader:
    def __init__(self, root_path, batch_size=12, frac_train_to_be_val=0.2, transform=None, target_transform=None):
        self.datasets = {}
        self.loaders = {}
        imgs_path = lambda x: f'{root_path}/{x}/Stimuli/'
        maps_path = lambda x: f'{root_path}/{x}/FIXATIONMAPS/'
        
        self.datasets['validation'] = CustomImageDataset(f'{root_path}/selected/images/', f'{root_path}/selected/maps/', 
                                                         transform=transform, target_transform=target_transform)
        self.datasets['test'] = CustomImageDataset(imgs_path('test'), transform=transform)
        dataset = CustomImageDataset(imgs_path('train'), maps_path('train'), transform=transform, target_transform=target_transform)
        splitter = DatasetValidationSplitter(len(dataset), frac_train_to_be_val)
        self.datasets['val'] = splitter.get_val_dataset(dataset)
        self.datasets['train'] = splitter.get_train_dataset(dataset)
        
        self.loaders['train'] = DataLoader(self.datasets['train'], batch_size=batch_size, shuffle = True, pin_memory=True)
        self.loaders['val'] = DataLoader(self.datasets['val'], batch_size=batch_size, shuffle = True, pin_memory=True)
        self.loaders['test'] = DataLoader(self.datasets['test'], batch_size=batch_size, shuffle = False, pin_memory=True)
        self.loaders['validation'] = DataLoader(self.datasets['validation'], batch_size=batch_size, shuffle = False, pin_memory=True)


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path, fix_maps_path=None, transform=None, target_transform=None):
        self.images = [os.path.join(imgs_path, category,img) for category in os.listdir(imgs_path)
                                 for img in os.listdir(os.path.join(imgs_path, category)) if img.endswith('.jpg')]
        self.maps = [os.path.join(fix_maps_path, category,img) for category in os.listdir(fix_maps_path)
                                 for img in os.listdir(os.path.join(fix_maps_path, category)) if img.endswith('.jpg')] if fix_maps_path else None
        self.transform = transform
        self.target_transform = target_transform
#         self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pil_img = Image.open(self.images[idx])
        image = self.transform(pil_img)
        # some images only have one channel
        if image.shape[0] == 1:
            image = image.expand(3, image.shape[1], image.shape[2]).clone()

        if self.maps:
            pil_fix_map = Image.open(self.maps[idx])
            fix_map = self.target_transform(pil_fix_map)
            
        return {
            'image': torch.as_tensor(image.clone()).float().contiguous(),
            'mask': torch.as_tensor(fix_map.clone()).float().contiguous(),
            'categories': self.images[idx].split('/')[-2]
        }