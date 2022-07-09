import cv2
import numpy as np 

def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


class Cat200Loader:
    def __init__(self, root_path, batch_size=8, frac_train_to_be_val=0.2):
        self.datasets = {}
        self.loaders = {}
        imgs_path = lambda x: f'{root_path}/{x}/Stimuli/'
        maps_path = lambda x: f'{root_path}/{x}/FIXATIONMAPS/'
        
        self.datasets['test'] = CustomImageDataset(imgs_path('test'), transform=transform1)
        dataset = CustomImageDataset(imgs_path('train'), maps_path('train'), transform=transform1, target_transform=transform2)
        splitter = DatasetValidationSplitter(len(dataset), frac_train_to_be_val)
        self.datasets['val'] = splitter.get_val_dataset(dataset)
        self.datasets['train'] = splitter.get_train_dataset(dataset)
        
        self.loaders['train'] = DataLoader(self.datasets['train'], batch_size=batch_size, shuffle = True, pin_memory=True)
        self.loaders['val'] = DataLoader(self.datasets['val'], batch_size=batch_size, shuffle = True, pin_memory=True)
        self.loaders['test'] = DataLoader(self.datasets['test'], batch_size=batch_size, shuffle = False, pin_memory=True)
        
        
class CustomImageDataset(Dataset):
    def __init__(self, imgs_path, fix_maps_path=None, transform=None, target_transform=None):
        self.images = [os.path.join(imgs_path, category,img) for category in os.listdir(imgs_path)
                                 for img in os.listdir(os.path.join(imgs_path, category)) if img.endswith('.jpg')]
        self.maps = [os.path.join(fix_maps_path, category,img) for category in os.listdir(fix_maps_path)
                                 for img in os.listdir(os.path.join(fix_maps_path, category)) if img.endswith('.jpg')] if fix_maps_path else None
        self.transform = transform
        self.target_transform = target_transform
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = padding(image, image_size1, image_size2, 3).astype('float')
        image = np.rollaxis(image, 2, 0)  
        if self.maps:
            fix_map = cv2.imread(self.maps[idx],0)
            fix_map = padding(fix_map, shape_r_gt, shape_c_gt, 1).astype('float')
        if self.transform:
            image = torch.tensor(image,dtype=torch.float)
            if image.shape[0] == 1:
                image = image.expand(3,image_size1,image_size2)
            image = self.norm(image)
            if self.maps:
                fix_map = torch.tensor(fix_map,dtype=torch.float)
                fix_map = fix_map.repeat(1,8,8)
        
        catt = torch.cat([image, fix_map], 0)
        return catt / 255.0, self.images[idx], self.maps[idx]