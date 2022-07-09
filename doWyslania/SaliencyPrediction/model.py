import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class MLNet(nn.Module):
    
    def __init__(self,prior_size, last_layer_to_freeze=23):
        super(MLNet, self).__init__()
        features = list(models.vgg16(pretrained = True).features)[:-1]
        features[23].stride = 1
        features[23].kernel_size = 5
        features[23].padding = 2
                
        self.features = nn.ModuleList(features).eval()
        for i,param in enumerate(self.features.parameters()):
            if i < last_layer_to_freeze:
                param.requires_grad = False
        
        self.dropout = nn.Dropout2d(p=0.5)
        self.saliency_conv = nn.Conv2d(1280,64,kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.final_conv = nn.Conv2d(64,1,kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.prior = nn.Parameter(torch.ones((1, 1, *prior_size), requires_grad=True))
        self.biupsc = torch.nn.UpsamplingBilinear2d(scale_factor=10)
        
    def forward(self, x):
        results = []
        for i,model in enumerate(self.features):
            x = model(x)
            if i in {16,23,29}:
                results.append(x)
        
        x = torch.cat(results, axis=1) 
        x = self.dropout(x)
        x = self.saliency_conv(x)
        x = self.final_conv(x)
        return F.relu(x * self.biupsc(self.prior))
