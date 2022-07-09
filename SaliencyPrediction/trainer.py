import gc
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from livelossplot import PlotLosses
from torchvision.utils import save_image
from torchvision.transforms import Resize


class Trainer:
    def __init__(self, model, criterion, optimizer, loaders):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.loaders = loaders
        
    def run_trainer(self, epochs):
        liveloss = PlotLosses()
        for epoch in range(epochs):
            self.logs = {}
            
            self.model.train()
            self.run_epoch('train', epoch)
            
            self.model.eval()
            with torch.no_grad():
                self.run_epoch('val', epoch)
                self.run_epoch('validation', epoch)
                
            liveloss.update(self.logs)
            liveloss.send()
            gc.collect()
                
    def run_epoch(self, phase, epoch):
        i = 0
        running_loss = 0.0
        for data in tqdm(self.loaders.loaders[phase]):
            x_true, y_true = data['image'], data['mask']
            x_true, y_true = x_true.to(self.device), y_true.to(self.device)
            y_pred = self.model(x_true)
            loss = self.criterion(y_pred, y_true, self.model.prior.clone())
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            running_loss += loss.detach() * x_true.size(0)
            if i % 10 == 0:
                print(f'loss on the given batch: {loss.item():.4f}')
                
            if phase == 'validation':
                for i, fix_map in enumerate(y_pred):
                    path_to_save = f'cat2000/selected/maps/{data["categories"][i]}/outputs/{epoch}.jpeg'
                    res = Resize((480, 640))
                    save_image(res(fix_map), path_to_save)
                    
            i += 1
            
        if phase != 'validation':  
            epoch_loss = running_loss / len(self.loaders.loaders[phase].dataset)
            self.logs[f'{phase}_loss'] = epoch_loss.item()
        
    def validation(self, epoch):
        for data in tqdm(self.loaders.loaders['validation']):
            x_true, y_true = data['image'], data['mask']
            x_true, y_true = x_true.to(self.device), y_true.to(self.device)
            print(x_true.shape, y_true.shape)
            y_pred = self.model(x_true)
            for i, fix_map in enumerate(y_pred):
                path_to_save = f'cat2000/selected/maps/{data["categories"][i]}/outputs/{epoch}.jpeg'
                res = Resize((480, 640))
                save_image(res(fix_map), path_to_save)
#                 plt.imshow(fix_map[0].data.cpu().numpy(),cmap='gray')
#                 plt.show()
                